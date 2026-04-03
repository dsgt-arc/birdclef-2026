#!/usr/bin/env python3
"""Build our own perch-meta dataset (equivalent to jaejohn/perch-meta).

Extracts Perch v2 embeddings + mapped logits for all fully-labeled soundscape
files and saves in the format expected by the pantanal-distill pipeline:
  - full_perch_meta.parquet: row_id, filename, site, hour_utc
  - full_perch_arrays.npz: scores_full_raw (N, 234), emb_full (N, 1536)

This replaces the jaejohn/perch-meta dependency with our own dataset.

Usage:
    python scripts/build_perch_meta.py
    python scripts/build_perch_meta.py --output-dir /path/to/output
    python scripts/build_perch_meta.py --use-onnx  # Use ONNX instead of TF
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# Defaults
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
TF_MODEL_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu")
ONNX_MODEL_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-onnx")
ARTIFACTS_DIR = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
)
OUTPUT_DIR = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/datasets/birdclef-2026/perch-meta"
)

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12


def parse_soundscape_metadata(filename: str) -> tuple:
    """Parse site and hour from soundscape filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None
    site = parts[3]
    try:
        hour = int(parts[5][:2])
    except (ValueError, IndexError):
        return site, None
    return site, hour


def get_fully_labeled_files(data_dir: Path) -> list[str]:
    """Get list of fully-labeled soundscape files (all 12 windows annotated)."""
    labels_path = data_dir / "train_soundscapes_labels.csv"
    labels = pd.read_csv(labels_path)

    # Count windows per file
    labels["filename"] = labels["row_id"].apply(
        lambda x: "_".join(x.rsplit("_", 1)[0].split("_")) + ".ogg"
    )
    # Actually parse filename from row_id: "BC2026_Train_NNNN_SXX_DATE_TIME_END" -> filename
    labels["filename"] = labels["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]) + ".ogg")
    windows_per_file = labels.groupby("filename").size()
    fully_labeled = windows_per_file[windows_per_file == N_WINDOWS].index.tolist()
    return sorted(fully_labeled)


def load_and_window_audio(audio_path: Path) -> np.ndarray:
    """Load a 60s soundscape and split into 12 x 5s windows."""
    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    expected_samples = 60 * SR
    if len(audio) < expected_samples:
        audio = np.pad(audio, (0, expected_samples - len(audio)))
    elif len(audio) > expected_samples:
        audio = audio[:expected_samples]

    windows = []
    for w in range(N_WINDOWS):
        start = w * WINDOW_SAMPLES
        windows.append(audio[start : start + WINDOW_SAMPLES])
    return np.stack(windows, axis=0).astype(np.float32)


def compute_mapped_scores(logits: np.ndarray, species_mapping: dict) -> np.ndarray:
    """Map raw Perch logits (14795) to BirdCLEF species scores (234) in logit space."""
    birdclef_species = species_mapping["birdclef_species"]
    direct_map = species_mapping["direct_map"]
    genus_proxy_map = species_mapping["genus_proxy_map"]
    n_species = len(birdclef_species)
    n_samples = logits.shape[0]
    n_logits = logits.shape[1]

    # Map in logit space (not sigmoid) — matching pantanal-distill pipeline
    scores = np.full((n_samples, n_species), -10.0, dtype=np.float32)

    for i, label in enumerate(birdclef_species):
        if label in direct_map:
            model_idx = direct_map[label] - 1
            if 0 <= model_idx < n_logits:
                scores[:, i] = logits[:, model_idx]
        elif label in genus_proxy_map:
            proxy_indices = [idx - 1 for idx in genus_proxy_map[label] if idx - 1 >= 0]
            valid = [idx for idx in proxy_indices if idx < n_logits]
            if valid:
                scores[:, i] = logits[:, valid].max(axis=1)
        # else: unmapped → stays -10.0

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--use-onnx", action="store_true", help="Use ONNX instead of TF")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building perch-meta dataset")
    print("=" * 60)

    # Load species mapping
    mapping_path = ARTIFACTS_DIR / "perch_species_mapping.json"
    with open(mapping_path) as f:
        species_mapping = json.load(f)
    print(f"Species: {len(species_mapping['birdclef_species'])}")

    # Get fully-labeled files
    fully_labeled = get_fully_labeled_files(DATA_DIR)
    print(f"Fully-labeled files: {len(fully_labeled)}")

    soundscape_dir = DATA_DIR / "train_soundscapes"

    # Load inference backend
    if args.use_onnx:
        import onnxruntime as ort

        onnx_path = ONNX_MODEL_DIR / "perch_v2.onnx"
        print(f"Using ONNX: {onnx_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

        def run_inference(waveforms):
            outputs = session.run(None, {"inputs": waveforms.astype(np.float32)})
            return outputs[0].astype(np.float32), outputs[3].astype(np.float32)
    else:
        import tensorflow as tf

        model = tf.saved_model.load(str(TF_MODEL_DIR))
        if hasattr(model, "infer_tf"):
            infer_fn = model.infer_tf
        else:
            infer_fn = model.signatures["serving_default"]

        def run_inference(waveforms):
            output = infer_fn(inputs=tf.constant(waveforms, dtype=tf.float32))
            emb = logits = None
            for _k, v in output.items():
                arr = v.numpy()
                if arr.shape[-1] == 1536:
                    emb = arr
                elif arr.shape[-1] > 10000:
                    logits = arr
            return emb.astype(np.float32), logits.astype(np.float32)

    # Process files
    all_meta = []
    all_scores = []
    all_emb = []

    t_start = time.time()
    for file_idx, filename in enumerate(fully_labeled):
        audio_path = soundscape_dir / filename
        if not audio_path.exists():
            print(f"  WARNING: {filename} not found, skipping")
            continue

        waveforms = load_and_window_audio(audio_path)
        embeddings, logits = run_inference(waveforms)
        scores = compute_mapped_scores(logits, species_mapping)

        site, hour = parse_soundscape_metadata(filename)

        for w in range(N_WINDOWS):
            end_time = (w + 1) * WINDOW_SEC
            stem = Path(filename).stem
            row_id = f"{stem}_{end_time}"
            all_meta.append(
                {
                    "row_id": row_id,
                    "filename": filename,
                    "site": site,
                    "hour_utc": hour,
                }
            )

        all_scores.append(scores)
        all_emb.append(embeddings)

        elapsed = time.time() - t_start
        print(f"  [{file_idx + 1}/{len(fully_labeled)}] {filename}: {elapsed:.1f}s total")

    # Stack arrays
    scores_full_raw = np.concatenate(all_scores, axis=0)
    emb_full = np.concatenate(all_emb, axis=0)
    meta_full = pd.DataFrame(all_meta)

    print("\nResults:")
    print(f"  meta_full: {meta_full.shape}")
    print(f"  scores_full_raw: {scores_full_raw.shape} {scores_full_raw.dtype}")
    print(f"  emb_full: {emb_full.shape} {emb_full.dtype}")

    # Save in jaejohn/perch-meta compatible format
    meta_path = output_dir / "full_perch_meta.parquet"
    npz_path = output_dir / "full_perch_arrays.npz"

    meta_full.to_parquet(meta_path, index=False)
    np.savez_compressed(npz_path, scores_full_raw=scores_full_raw, emb_full=emb_full)

    print("\nSaved:")
    print(f"  {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also create dataset-metadata.json for Kaggle upload
    dataset_meta = {
        "title": "BirdCLEF 2026 Perch Meta (our extraction)",
        "id": f"{ARTIFACTS_DIR.name}/birdclef-2026-perch-meta-v1",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_json_path = output_dir / "dataset-metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(dataset_meta, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}min)")


if __name__ == "__main__":
    main()
