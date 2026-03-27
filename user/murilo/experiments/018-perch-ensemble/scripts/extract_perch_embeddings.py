"""Extract Perch v2 embeddings and logits from soundscapes.

Loads the Perch v2 TF SavedModel and processes soundscape files.

Splits:
  - labeled (default): filenames from train_soundscapes_labels.csv
  - unlabeled: all files in soundscape_dir NOT in labels CSV

Outputs (saved to config paths.embeddings_dir / <split>):
  - perch_embeddings.parquet (N rows x 1536 embedding dims)
  - perch_logits.parquet (N rows x 14795 Perch logit dims)
  - perch_mapped_scores.parquet (N rows x 234 BirdCLEF species scores)

Requires: tensorflow, librosa, pandas, numpy, pyyaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Perch v2 embeddings and logits from soundscapes"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/perch_ensemble_v1.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="labeled",
        choices=["labeled", "unlabeled"],
        help="Which split to extract: labeled (from labels CSV) or unlabeled (remaining files)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override Perch v2 TF SavedModel directory from config",
    )
    parser.add_argument(
        "--soundscape-dir",
        type=str,
        default=None,
        help="Override train_soundscapes audio directory from config",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Override train_soundscapes_labels.csv path from config",
    )
    parser.add_argument(
        "--species-mapping",
        type=str,
        default=None,
        help="Override perch_species_mapping.json path from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config (default: config embeddings_dir)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Target sample rate for Perch v2 (default: 32000)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Window size in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for inference (default: 12, one full 60s file)",
    )
    return parser.parse_args()


def load_perch_model(model_dir: str):
    """Load Perch v2 TF SavedModel and discover inference signature."""
    import tensorflow as tf

    print(f"Loading Perch v2 model from: {model_dir}")
    model = tf.saved_model.load(model_dir)

    # Discover available signatures
    print(f"  Available signatures: {list(model.signatures.keys())}")

    # Try common inference methods
    infer_fn = None

    # Method 1: Check for 'infer_tf' method (common for Perch)
    if hasattr(model, "infer_tf"):
        infer_fn = model.infer_tf
        print("  Using model.infer_tf() for inference")
    # Method 2: Check for 'serving_default' signature
    elif "serving_default" in model.signatures:
        infer_fn = model.signatures["serving_default"]
        print("  Using model.signatures['serving_default'] for inference")
    # Method 3: Try calling model directly
    elif callable(model):
        infer_fn = model
        print("  Using model() directly for inference")
    else:
        # Method 4: Use first available signature
        sig_keys = list(model.signatures.keys())
        if sig_keys:
            infer_fn = model.signatures[sig_keys[0]]
            print(f"  Using model.signatures['{sig_keys[0]}'] for inference")
        else:
            raise RuntimeError(
                "Could not find inference method on Perch model. "
                f"Available attributes: {dir(model)}"
            )

    return model, infer_fn


def run_inference(infer_fn, waveform_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run Perch v2 inference on a batch of waveforms.

    Args:
        infer_fn: TF inference function
        waveform_batch: (batch, samples) float32 waveform at 32kHz

    Returns:
        embeddings: (batch, 1536) float32
        logits: (batch, 14795) float32
    """
    import tensorflow as tf

    input_tensor = tf.constant(waveform_batch, dtype=tf.float32)
    output = infer_fn(input_tensor)

    # Handle different output formats
    if isinstance(output, dict):
        # Common keys: 'embedding', 'logits', 'output_0', 'output_1'
        embedding_keys = ["embedding", "embeddings", "output_0"]
        logit_keys = ["logits", "logit", "output_1"]

        embeddings = None
        logits = None

        for key in embedding_keys:
            if key in output:
                embeddings = output[key].numpy()
                break

        for key in logit_keys:
            if key in output:
                logits = output[key].numpy()
                break

        if embeddings is None or logits is None:
            # Fall back: identify by shape
            print(f"  Output keys: {list(output.keys())}")
            for key, val in output.items():
                arr = val.numpy()
                print(f"    {key}: shape={arr.shape}")
                if arr.shape[-1] == 1536 and embeddings is None:
                    embeddings = arr
                elif arr.shape[-1] > 10000 and logits is None:
                    logits = arr

        if embeddings is None or logits is None:
            raise RuntimeError(
                f"Could not identify embeddings/logits from output keys: {list(output.keys())}"
            )
    elif isinstance(output, tuple | list):
        # Assume (embeddings, logits) or (logits, embeddings)
        arr0 = output[0].numpy() if hasattr(output[0], "numpy") else np.array(output[0])
        arr1 = output[1].numpy() if hasattr(output[1], "numpy") else np.array(output[1])

        if arr0.shape[-1] == 1536:
            embeddings, logits = arr0, arr1
        else:
            logits, embeddings = arr0, arr1
    else:
        raise RuntimeError(f"Unexpected output type: {type(output)}")

    return embeddings, logits


def compute_mapped_scores(
    logits: np.ndarray,
    species_mapping: dict,
) -> np.ndarray:
    """Map Perch logits (14795) to BirdCLEF species scores (234).

    Direct matches: sigmoid(logit[perch_idx])
    Genus proxies: max(sigmoid(logits[proxy_indices]))
    Unmapped: 0.0
    """
    birdclef_species = species_mapping["birdclef_species"]
    direct_map = species_mapping["direct_map"]
    genus_proxy_map = species_mapping["genus_proxy_map"]
    n_species = len(birdclef_species)
    n_samples = logits.shape[0]

    # Compute sigmoid of all logits once
    sigmoid_logits = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    sigmoid_logits = sigmoid_logits.astype(np.float32)

    scores = np.zeros((n_samples, n_species), dtype=np.float32)

    for i, label in enumerate(birdclef_species):
        if label in direct_map:
            perch_idx = direct_map[label]
            scores[:, i] = sigmoid_logits[:, perch_idx]
        elif label in genus_proxy_map:
            proxy_indices = genus_proxy_map[label]
            scores[:, i] = sigmoid_logits[:, proxy_indices].max(axis=1)
        # else: unmapped -> stays 0.0

    return scores


def main() -> None:
    args = parse_args()

    # Resolve config
    experiment_dir = Path(__file__).resolve().parent.parent
    if args.config is None:
        config_path = experiment_dir / "config" / "perch_ensemble_v1.yaml"
    else:
        config_path = Path(args.config)

    config = load_config(config_path)
    paths = config["paths"]

    # Resolve paths (CLI overrides > config)
    model_dir = args.model_dir or paths["perch_model_dir"]
    soundscape_dir = args.soundscape_dir or paths["soundscape_dir"]
    labels_csv = args.labels_csv or paths["soundscape_labels_csv"]

    if args.species_mapping is not None:
        species_mapping_path = Path(args.species_mapping)
    else:
        species_mapping_path = (
            experiment_dir / paths["local_artifacts_dir"] / "perch_species_mapping.json"
        )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        # Config embeddings_dir already includes /labeled; strip it and re-append split
        base_embeddings_dir = Path(paths["embeddings_dir"]).parent
        output_dir = base_embeddings_dir / args.split

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:  {config_path}")
    print(f"Split:   {args.split}")
    print(f"Output:  {output_dir}")

    # Load species mapping
    print(f"\nLoading species mapping from: {species_mapping_path}")
    with open(species_mapping_path) as f:
        species_mapping = json.load(f)

    birdclef_species = species_mapping["birdclef_species"]
    print(f"  BirdCLEF species: {len(birdclef_species)}")

    # Get filenames based on split
    labels_df = pd.read_csv(labels_csv)
    labeled_filenames = set(labels_df["filename"].unique())

    if args.split == "labeled":
        filenames = sorted(labeled_filenames)
        print(f"\nFound {len(filenames)} labeled soundscape files from labels CSV")
    else:
        # unlabeled: all files in soundscape_dir NOT in labels CSV
        all_files = {f.name for f in Path(soundscape_dir).iterdir() if f.suffix == ".ogg"}
        unlabeled_filenames = all_files - labeled_filenames
        filenames = sorted(unlabeled_filenames)
        print(
            f"\nFound {len(all_files)} total files, {len(labeled_filenames)} labeled, "
            f"{len(filenames)} unlabeled"
        )

    # Load Perch model
    model, infer_fn = load_perch_model(model_dir)

    # Probe model with dummy input to confirm shapes
    window_samples = int(args.sample_rate * args.window_sec)
    dummy = np.zeros((1, window_samples), dtype=np.float32)
    embeddings_test, logits_test = run_inference(infer_fn, dummy)
    print("\nModel output shapes (from dummy input):")
    print(f"  Embeddings: {embeddings_test.shape}")
    print(f"  Logits: {logits_test.shape}")

    embedding_dim = embeddings_test.shape[-1]
    n_perch_classes = logits_test.shape[-1]

    # Process all soundscape files
    all_embeddings = []
    all_logits = []
    all_metadata = []

    n_windows_per_file = int(60.0 / args.window_sec)  # 12 windows for 60s / 5s
    print(f"\nProcessing {len(filenames)} files, {n_windows_per_file} windows each...")
    print(f"  Window: {args.window_sec}s at {args.sample_rate}Hz = {window_samples} samples")

    t_start = time.time()

    for file_idx, filename in enumerate(filenames):
        audio_path = os.path.join(soundscape_dir, filename)

        if not os.path.exists(audio_path):
            print(f"  WARNING: File not found: {audio_path}, skipping")
            continue

        # Load full 60s audio at 32kHz
        audio, sr = librosa.load(audio_path, sr=args.sample_rate, mono=True)

        # Ensure we have exactly 60s worth of samples
        expected_samples = int(60.0 * args.sample_rate)
        if len(audio) < expected_samples:
            # Pad with zeros if shorter
            audio = np.pad(audio, (0, expected_samples - len(audio)))
        elif len(audio) > expected_samples:
            audio = audio[:expected_samples]

        # Split into windows
        windows = []
        for w in range(n_windows_per_file):
            start_sample = w * window_samples
            end_sample = start_sample + window_samples
            windows.append(audio[start_sample:end_sample])

        waveform_batch = np.stack(windows, axis=0).astype(np.float32)

        # Run inference
        embeddings, logits = run_inference(infer_fn, waveform_batch)

        all_embeddings.append(embeddings)
        all_logits.append(logits)

        # Build metadata for each window
        for w in range(n_windows_per_file):
            window_start = w * args.window_sec
            window_end = window_start + args.window_sec
            all_metadata.append(
                {
                    "filename": filename,
                    "window_idx": w,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )

        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            elapsed = time.time() - t_start
            rate = (file_idx + 1) / elapsed
            eta = (len(filenames) - file_idx - 1) / rate
            print(
                f"  [{file_idx + 1}/{len(filenames)}] "
                f"{filename} -> {embeddings.shape[0]} windows | "
                f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s"
            )

    elapsed_total = time.time() - t_start
    print(f"\nInference complete: {len(all_metadata)} windows in {elapsed_total:.1f}s")

    # Stack all results
    all_embeddings_arr = np.concatenate(all_embeddings, axis=0)
    all_logits_arr = np.concatenate(all_logits, axis=0)

    print(f"  Embeddings shape: {all_embeddings_arr.shape}")
    print(f"  Logits shape: {all_logits_arr.shape}")

    # Compute mapped scores
    print("\nComputing mapped BirdCLEF species scores...")
    mapped_scores = compute_mapped_scores(all_logits_arr, species_mapping)
    print(f"  Mapped scores shape: {mapped_scores.shape}")

    # Build DataFrames
    metadata_df = pd.DataFrame(all_metadata)

    # Embeddings parquet
    embedding_cols = [f"emb_{i}" for i in range(embedding_dim)]
    embeddings_df = pd.concat(
        [
            metadata_df,
            pd.DataFrame(all_embeddings_arr, columns=embedding_cols),
        ],
        axis=1,
    )

    # Logits parquet
    logit_cols = [f"logit_{i}" for i in range(n_perch_classes)]
    logits_df = pd.concat(
        [
            metadata_df,
            pd.DataFrame(all_logits_arr, columns=logit_cols),
        ],
        axis=1,
    )

    # Mapped scores parquet
    mapped_df = pd.concat(
        [
            metadata_df,
            pd.DataFrame(mapped_scores, columns=birdclef_species),
        ],
        axis=1,
    )

    # Save
    emb_path = output_dir / "perch_embeddings.parquet"
    logit_path = output_dir / "perch_logits.parquet"
    mapped_path = output_dir / "perch_mapped_scores.parquet"

    embeddings_df.to_parquet(emb_path, index=False)
    logits_df.to_parquet(logit_path, index=False)
    mapped_df.to_parquet(mapped_path, index=False)

    print("\nSaved outputs:")
    print(f"  Embeddings: {emb_path} ({emb_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Logits:     {logit_path} ({logit_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Mapped:     {mapped_path} ({mapped_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Quick sanity check
    n_direct = len(species_mapping["direct_map"])
    n_genus = len(species_mapping["genus_proxy_map"])
    n_unmapped = len(species_mapping["unmapped"])
    mapped_nonzero = (mapped_scores.sum(axis=0) > 0).sum()
    print(f"\n  Sanity check: {mapped_nonzero} species have non-zero scores")
    print(f"    Expected: {n_direct + n_genus} (direct + genus proxy)")
    print(f"    Unmapped (zero scores): {n_unmapped}")


if __name__ == "__main__":
    main()
