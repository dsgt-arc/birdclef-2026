#!/usr/bin/env python3
"""BirdCLEF 2026 Perch Cache Kernel (Notebook A).

Runs Perch v2 inference on all test soundscapes and saves embeddings + mapped
scores as kernel output. A downstream scoring kernel (Notebook B) loads these
cached results and applies probes + priors + smoothing + ensemble blending.

This two-notebook chain is necessary because Perch inference on ~600 test
soundscapes takes ~80 min, leaving no budget for post-processing or SED
inference within a single 90 min kernel.

Output files (saved to /kaggle/working/):
  - perch_cache.npz: embeddings (N,1536) + mapped_scores (N,234)
  - perch_meta.parquet: soundscape_id, end_time, site, hour, file_idx, window_idx
"""

# ---------------------------------------------------------------------------
# Offline dependency installation (runs on Kaggle before imports)
# ---------------------------------------------------------------------------
import os
import subprocess
import sys
import time

START_TIME = time.time()
# Safety cutoff: stop processing new files if we're within 2 min of the 90-min limit.
# We still need time to save outputs.
TIMEOUT_SECONDS = 88 * 60  # 88 min

KAGGLE_USER = "murilogustineli"


def _find_input_early(slug: str) -> str:
    """Resolve Kaggle input path, supporting both old and new directory layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # fallback to old-style


# Debug: show what we see in /kaggle/input
if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")

# Install TensorFlow 2.20 from pre-built wheel (Kaggle's default TF is too old for Perch v2)
TF_WHEEL_CANDIDATES = [
    _find_input_early("birdclef-tf220-wheel"),
    "/kaggle/input/birdclef-tf220-wheel",
    f"/kaggle/input/datasets/{KAGGLE_USER}/birdclef-tf220-wheel",
]
for tf_wheel_dir in TF_WHEEL_CANDIDATES:
    if os.path.exists(tf_wheel_dir):
        print(f"Installing TensorFlow 2.20 from {tf_wheel_dir}")
        for whl in sorted(os.listdir(tf_wheel_dir)):
            if whl.endswith(".whl"):
                whl_path = os.path.join(tf_wheel_dir, whl)
                print(f"  Installing {whl}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", whl_path])
        break
else:
    print("WARNING: TF 2.20 wheel not found, using Kaggle default TensorFlow")

# ---------------------------------------------------------------------------
# Imports (after offline install)
# ---------------------------------------------------------------------------
import json  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

try:
    import librosa  # noqa: E402
except ImportError:
    librosa = None  # only needed for resampling non-32kHz audio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERCH_SR = 32000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SECONDS  # 160000
SOUNDSCAPE_SECONDS = 60
N_WINDOWS = SOUNDSCAPE_SECONDS // WINDOW_SECONDS  # 12
N_SPECIES_2026 = 234

# Perch ensemble artifacts slug (for species mapping)
PERCH_ARTIFACTS_SLUG = "birdclef-2026-perch-ensemble-v1"

# Perch v2 model candidates
PERCH_MODEL_CANDIDATES = [
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier",
]


# ===========================================================================
# Path resolution
# ===========================================================================


def _find_input(slug: str) -> str | None:
    """Resolve Kaggle input path, supporting both old and new layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_perch_model_dir() -> Path | None:
    """Find Perch v2 TF SavedModel directory.

    Kaggle mounts model sources at paths that vary by layout version.
    We search candidates and also walk /kaggle/input/ to find saved_model.pb.
    """
    # First try explicit candidates
    for candidate in PERCH_MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            if (p / "saved_model.pb").exists():
                return p
            # Check one level deeper
            for subdir in sorted(p.iterdir()):
                if subdir.is_dir() and (subdir / "saved_model.pb").exists():
                    return subdir
            return p

    # Fallback: walk /kaggle/input/ looking for saved_model.pb
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("\n  DEBUG: Walking /kaggle/input/ to find Perch saved_model.pb...")
        for depth1 in sorted(kaggle_input.iterdir()):
            if depth1.name.startswith("."):
                continue
            if (depth1 / "saved_model.pb").exists():
                return depth1
            if depth1.is_dir():
                for depth2 in sorted(depth1.iterdir()):
                    if (depth2 / "saved_model.pb").exists():
                        return depth2
                    if depth2.is_dir():
                        for depth3 in sorted(depth2.iterdir()):
                            if (depth3 / "saved_model.pb").exists():
                                return depth3
                            if depth3.is_dir():
                                for depth4 in sorted(depth3.iterdir()):
                                    if (depth4 / "saved_model.pb").exists():
                                        return depth4

        # Print directory listing for debugging
        print("  DEBUG: saved_model.pb not found. Directory listing:")
        for depth1 in sorted(kaggle_input.iterdir()):
            print(f"    {depth1.name}/")
            if depth1.is_dir():
                for depth2 in sorted(depth1.iterdir()):
                    print(f"      {depth2.name}/")
                    if depth2.is_dir():
                        for depth3 in sorted(depth2.iterdir()):
                            print(f"        {depth3.name}/")

    # Local testing fallback
    local = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu")
    if local.exists():
        return local
    local2 = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2")
    if local2.exists():
        return local2

    return None


def find_perch_artifacts_dir() -> Path | None:
    """Find Perch ensemble artifacts directory (for species mapping)."""
    path = _find_input(PERCH_ARTIFACTS_SLUG)
    if path and os.path.exists(path):
        return Path(path)

    # Local testing fallback
    local = Path("artifacts")
    if (local / "perch_species_mapping.json").exists():
        return local

    # Shared storage fallback
    shared = Path(
        "/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
    )
    if shared.exists():
        return shared

    return None


def find_competition_dir() -> Path:
    """Resolve competition data directory."""
    candidates = [
        "/kaggle/input/competitions/birdclef-2026",
        "/kaggle/input/birdclef-2026",
    ]
    for c in candidates:
        if os.path.exists(c):
            return Path(c)
    return Path(os.environ.get("COMPETITION_DIR", "data/birdclef-2026"))


# ===========================================================================
# Perch v2 inference
# ===========================================================================


def load_perch_model(model_dir: str):
    """Load Perch v2 TF SavedModel and discover inference signature.

    Returns (model, infer_fn) tuple.
    """
    import tensorflow as tf

    print(f"  Loading Perch v2 model from: {model_dir}")
    model = tf.saved_model.load(str(model_dir))

    # Discover available signatures
    sigs = list(model.signatures.keys()) if hasattr(model, "signatures") else []
    print(f"  Available signatures: {sigs}")

    infer_fn = None

    if hasattr(model, "infer_tf"):
        infer_fn = model.infer_tf
        print("  Using model.infer_tf() for inference")
    elif "serving_default" in sigs:
        infer_fn = model.signatures["serving_default"]
        print("  Using model.signatures['serving_default'] for inference")
    elif callable(model):
        infer_fn = model
        print("  Using model() directly for inference")
    else:
        if sigs:
            infer_fn = model.signatures[sigs[0]]
            print(f"  Using model.signatures['{sigs[0]}'] for inference")
        else:
            raise RuntimeError(
                "Could not find inference method on Perch model. "
                f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}"
            )

    return model, infer_fn


def run_perch_inference(infer_fn, waveform_batch):
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

    try:
        output = infer_fn(inputs=input_tensor)
    except TypeError:
        output = infer_fn(input_tensor)

    if isinstance(output, dict):
        embedding_keys = ["embedding", "embeddings", "output_0"]
        logit_keys = ["label", "logits", "logit", "output_1"]

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
            print(f"  Perch output keys: {list(output.keys())}")
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
        arr0 = output[0].numpy() if hasattr(output[0], "numpy") else np.array(output[0])
        arr1 = output[1].numpy() if hasattr(output[1], "numpy") else np.array(output[1])

        if arr0.shape[-1] == 1536:
            embeddings, logits = arr0, arr1
        else:
            logits, embeddings = arr0, arr1
    else:
        raise RuntimeError(f"Unexpected Perch output type: {type(output)}")

    return embeddings.astype(np.float32), logits.astype(np.float32)


def compute_perch_mapped_scores(logits, perch_species_mapping):
    """Map Perch logits to BirdCLEF species scores (234).

    The species mapping indices come from labels.csv line numbers where
    line 0 = 'inat2024_fsd50k' (non-bird catch-all). But Perch v2 outputs
    14,795 logits (dropping index 0), so model_output[i] = labels.csv[i+1].
    We subtract 1 from all mapping indices to align with model output.
    """
    birdclef_species = perch_species_mapping["birdclef_species"]
    direct_map = perch_species_mapping["direct_map"]
    genus_proxy_map = perch_species_mapping["genus_proxy_map"]
    n_species = len(birdclef_species)
    n_samples = logits.shape[0]
    n_logits = logits.shape[1]

    # Compute sigmoid of all logits once
    sigmoid_logits = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    sigmoid_logits = sigmoid_logits.astype(np.float32)

    scores = np.zeros((n_samples, n_species), dtype=np.float32)

    for i, label in enumerate(birdclef_species):
        if label in direct_map:
            model_idx = direct_map[label] - 1
            if 0 <= model_idx < n_logits:
                scores[:, i] = sigmoid_logits[:, model_idx]
        elif label in genus_proxy_map:
            proxy_model_indices = [idx - 1 for idx in genus_proxy_map[label] if idx - 1 >= 0]
            valid_indices = [idx for idx in proxy_model_indices if idx < n_logits]
            if valid_indices:
                scores[:, i] = sigmoid_logits[:, valid_indices].max(axis=1)
        # else: unmapped -> stays 0.0

    return scores


# ===========================================================================
# Audio loading
# ===========================================================================


def load_and_window_audio(audio_path, sr=PERCH_SR):
    """Load a 60s soundscape and split into 12 x 5s windows.

    Returns:
        waveform_batch: (12, 160000) float32
    """
    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if file_sr != sr:
        if librosa is not None:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        else:
            raise RuntimeError(
                f"Audio SR is {file_sr}, expected {sr}. Install librosa for resampling."
            )

    # Ensure exactly 60s
    expected_samples = SOUNDSCAPE_SECONDS * sr
    if len(audio) < expected_samples:
        audio = np.pad(audio, (0, expected_samples - len(audio)))
    elif len(audio) > expected_samples:
        audio = audio[:expected_samples]

    # Split into windows
    windows = []
    for w in range(N_WINDOWS):
        start = w * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        windows.append(audio[start:end])

    return np.stack(windows, axis=0).astype(np.float32)


# ===========================================================================
# Metadata parsing
# ===========================================================================


def _parse_soundscape_metadata(filename):
    """Parse site and hour from soundscape filename.

    Format: BC2026_{Train|Test}_NNNN_SXX_YYYYMMDD_HHMMSS.ogg
    Returns: (site, hour) or (None, None) if parsing fails.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None

    site = parts[3]  # e.g. "S08"
    try:
        hour = int(parts[5][:2])  # first 2 digits of HHMMSS
    except (ValueError, IndexError):
        return site, None

    return site, hour


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 70)
    print("BirdCLEF 2026 Perch Cache Kernel (Notebook A)")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------------
    competition_dir = find_competition_dir()
    test_dir = competition_dir / "test_soundscapes"
    output_dir = Path("/kaggle/working")

    # Local testing fallbacks
    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not output_dir.exists():
        output_dir = Path(os.environ.get("OUTPUT_DIR", "predictions"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find Perch model
    perch_model_dir = find_perch_model_dir()
    if perch_model_dir is None:
        raise FileNotFoundError("Perch v2 model not found!")

    # Find species mapping
    perch_artifacts_dir = find_perch_artifacts_dir()
    if perch_artifacts_dir is None:
        raise FileNotFoundError("Perch artifacts directory not found (need species mapping)!")

    print(f"\nCompetition data: {competition_dir}")
    print(f"Test soundscapes: {test_dir}")
    print(f"Perch model: {perch_model_dir}")
    print(f"Perch artifacts: {perch_artifacts_dir}")
    print(f"Output dir: {output_dir}")

    # -------------------------------------------------------------------
    # Load species mapping
    # -------------------------------------------------------------------
    perch_mapping_path = None
    for name in ["perch_species_mapping.json", "species_mapping.json"]:
        p = perch_artifacts_dir / name
        if p.exists():
            perch_mapping_path = p
            break

    if perch_mapping_path is None:
        raise FileNotFoundError(
            f"Species mapping not found in {perch_artifacts_dir}. "
            "Expected perch_species_mapping.json."
        )

    with open(perch_mapping_path) as f:
        perch_species_mapping = json.load(f)

    birdclef_species = perch_species_mapping["birdclef_species"]
    n_direct = len(perch_species_mapping.get("direct_map", {}))
    n_genus = len(perch_species_mapping.get("genus_proxy_map", {}))
    n_unmapped = len(perch_species_mapping.get("unmapped", []))
    print(f"Species mapping: {n_direct} direct, {n_genus} genus proxy, {n_unmapped} unmapped")

    # -------------------------------------------------------------------
    # Load Perch model
    # -------------------------------------------------------------------
    print("\n--- Loading Perch v2 Model ---")
    t_load = time.time()
    _, infer_fn = load_perch_model(str(perch_model_dir))

    # Probe with dummy input
    dummy = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    test_emb, test_logits = run_perch_inference(infer_fn, dummy)
    print(f"  Output shapes: embeddings={test_emb.shape}, logits={test_logits.shape}")
    print(f"  Model loaded in {time.time() - t_load:.1f}s")

    # -------------------------------------------------------------------
    # Find soundscape files
    # -------------------------------------------------------------------
    audio_files = sorted(
        list(test_dir.glob("*.ogg")) + list(test_dir.glob("*.wav")) + list(test_dir.glob("*.flac"))
    )

    # Fallback: use train_soundscapes for local validation
    if len(audio_files) == 0:
        train_dir = competition_dir / "train_soundscapes"
        if train_dir.exists():
            audio_files = sorted(train_dir.glob("*.ogg"))[:3]
            print(f"No test soundscapes found, using {len(audio_files)} train files for validation")

    print(f"\nProcessing {len(audio_files)} soundscapes from {test_dir}")

    # -------------------------------------------------------------------
    # Process each soundscape (threaded, matching perch-raw pattern)
    # -------------------------------------------------------------------
    # ThreadPoolExecutor overlaps audio I/O with Perch inference.
    # The perch-raw notebook uses this pattern and finishes ~600 files
    # in ~85 min. Without threading, we'd likely time out.

    def process_file(args):
        """Process a single soundscape file. Called from ThreadPoolExecutor."""
        file_idx, audio_path = args

        if time.time() - START_TIME > TIMEOUT_SECONDS:
            # Return sentinel: zeros for embeddings/scores
            return (
                file_idx,
                audio_path,
                np.zeros((N_WINDOWS, 1536), dtype=np.float32),
                np.zeros((N_WINDOWS, N_SPECIES_2026), dtype=np.float32),
                True,  # timed_out
            )

        waveform_batch = load_and_window_audio(audio_path)
        embeddings, logits = run_perch_inference(infer_fn, waveform_batch)
        mapped_scores = compute_perch_mapped_scores(logits, perch_species_mapping)

        return file_idx, audio_path, embeddings, mapped_scores, False

    all_embeddings = []
    all_mapped_scores = []
    meta_rows = []
    n_timed_out = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_file, enumerate(audio_files)):
            file_idx, audio_path, embeddings, mapped_scores, timed_out = result

            if timed_out:
                n_timed_out += 1
                continue

            soundscape_id = audio_path.stem
            site, hour = _parse_soundscape_metadata(str(audio_path))

            all_embeddings.append(embeddings)
            all_mapped_scores.append(mapped_scores)

            for w in range(N_WINDOWS):
                meta_rows.append(
                    {
                        "soundscape_id": soundscape_id,
                        "end_time": (w + 1) * WINDOW_SECONDS,
                        "site": site,
                        "hour": hour,
                        "file_idx": file_idx,
                        "window_idx": w,
                    }
                )

            n_done = len(all_embeddings)
            elapsed = time.time() - START_TIME
            if n_done % 50 == 0 or n_done == 1 or n_done == len(audio_files):
                remaining = TIMEOUT_SECONDS - elapsed
                print(
                    f"  [{n_done}/{len(audio_files)}] {audio_path.name}: "
                    f"{elapsed / 60:.1f}min elapsed, {remaining / 60:.1f}min remaining"
                )

    if n_timed_out > 0:
        print(f"\n*** TIMEOUT: {n_timed_out} files skipped ***")

    # -------------------------------------------------------------------
    # Save cache outputs
    # -------------------------------------------------------------------
    print("\n--- Saving Cache ---")

    if len(all_embeddings) == 0:
        print("WARNING: No files processed! Saving empty cache.")
        all_emb = np.zeros((0, 1536), dtype=np.float32)
        all_scores = np.zeros((0, N_SPECIES_2026), dtype=np.float32)
    else:
        all_emb = np.concatenate(all_embeddings, axis=0)
        all_scores = np.concatenate(all_mapped_scores, axis=0)

    # Save embeddings + mapped scores as compressed NPZ
    cache_path = output_dir / "perch_cache.npz"
    np.savez_compressed(
        cache_path,
        embeddings=all_emb,
        mapped_scores=all_scores,
    )
    print(f"  Saved {cache_path}: embeddings {all_emb.shape}, mapped_scores {all_scores.shape}")
    print(f"  File size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save metadata as parquet
    meta_df = pd.DataFrame(meta_rows)
    meta_path = output_dir / "perch_meta.parquet"
    meta_df.to_parquet(meta_path, index=False)
    print(f"  Saved {meta_path}: {len(meta_df)} rows")

    # -------------------------------------------------------------------
    # Build submission.csv (raw Perch scores, no probes/smoothing)
    # -------------------------------------------------------------------
    # Required so this kernel can be submitted to the competition (test
    # soundscapes are only available during competition submission).
    # The raw scores produce ~0.825 LB, matching the starter notebook.
    print("\n--- Building submission.csv (raw Perch) ---")
    sample_sub_path = competition_dir / "sample_submission.csv"

    all_predictions = []
    for row_meta, scores in zip(meta_rows, all_scores, strict=False):
        row_id = f"{row_meta['soundscape_id']}_{row_meta['end_time']}"
        row = {"row_id": row_id}
        for sp_idx, species in enumerate(birdclef_species):
            row[species] = float(scores[sp_idx])
        all_predictions.append(row)

    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        expected_columns = list(sample_sub.columns)
        species_columns = [c for c in expected_columns if c != "row_id"]

        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            submission = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
            for col in species_columns:
                if col not in submission.columns:
                    submission[col] = 0.0
                else:
                    submission[col] = submission[col].fillna(0.0)
        else:
            submission = sample_sub.copy()
            for col in species_columns:
                submission[col] = 0.0
        submission = submission[expected_columns]
    else:
        submission = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()

    sub_path = output_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Saved {sub_path}: {len(submission)} rows")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total_time = time.time() - START_TIME
    n_processed = len(all_embeddings)
    print("\n--- Summary ---")
    print(f"  Files processed: {n_processed}/{len(audio_files)}")
    print(f"  Total windows: {all_emb.shape[0]}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    if n_processed > 0:
        print(f"  Avg per file: {total_time / n_processed:.1f}s")


if __name__ == "__main__":
    main()
