#!/usr/bin/env python3
"""BirdCLEF 2026 Scoring Kernel (Notebook B).

Two-notebook chain, stage 2:
  Notebook A (cache): Perch v2 inference on test soundscapes → embeddings + mapped_scores
  Notebook B (this):  Train probes on labeled soundscapes → apply to cached test → submit

Probes (PCA + LogReg + metadata priors) are trained IN THIS NOTEBOOK on the 66
labeled soundscapes, using the same Perch model as Notebook A. This avoids
cross-environment sklearn version mismatches that caused the 0.500 LB failure.

Set PERCH_ONLY=True for Phase B (Perch-only validation).
Set PERCH_ONLY=False for Phase C (SED + Perch ensemble).
"""

# ---------------------------------------------------------------------------
# Offline dependency installation
# ---------------------------------------------------------------------------
import os
import subprocess
import sys
import time

START_TIME = time.time()

KAGGLE_USER = "murilogustineli"

# Mode: True = Phase B (Perch-only), False = Phase C (SED + Perch ensemble)
PERCH_ONLY = True


def _find_input_early(slug: str) -> str:
    """Resolve Kaggle input path, supporting both old and new directory layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# Debug
if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")
print(f"PERCH_ONLY mode: {PERCH_ONLY}")

# Install TensorFlow 2.20 (needed to run Perch on labeled soundscapes for probe training)
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
    print("WARNING: TF 2.20 wheel not found")

# Install onnxruntime only when SED is needed
if not PERCH_ONLY:
    DEPS_DIR = _find_input_early("birdclef-deps")
    if os.path.exists(DEPS_DIR):
        print(f"Installing onnxruntime from {DEPS_DIR}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                DEPS_DIR,
                "onnxruntime",
            ]
        )

# ---------------------------------------------------------------------------
# Imports (after offline install)
# ---------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

if not PERCH_ONLY:
    import onnxruntime as ort  # noqa: E402

# ---------------------------------------------------------------------------
# Constants and post-processing flags
# ---------------------------------------------------------------------------
ENABLE_TTA = True
ENABLE_SED_TEMPORAL_SMOOTHING = True
ENABLE_SED_TOPN = True

N_SPECIES_2026 = 234
PERCH_SR = 32000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SECONDS  # 160000
SOUNDSCAPE_SECONDS = 60
N_WINDOWS = SOUNDSCAPE_SECONDS // WINDOW_SECONDS  # 12

# Kaggle dataset slugs
PERCH_CACHE_SLUG = "birdclef-2026-perch-cache-v1"
PERCH_ARTIFACTS_SLUG = "birdclef-2026-perch-ensemble-v1"
SED_MODEL_SLUG = "birdclef-2026-effnet-b0-sed-v4b-weights"

# Perch v2 model candidates
PERCH_MODEL_CANDIDATES = [
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier",
]

# Probe training hyperparameters (matching config/perch_ensemble_v1.yaml)
PROBE_PCA_COMPONENTS = 32
PROBE_LOGREG_C = 0.25
PROBE_MIN_POSITIVE = 8
PROBE_ALPHA_BASE = 0.25
PROBE_ALPHA_SCALE = 0.25
PROBE_ALPHA_THRESHOLD = 40

# Prior hyperparameters
PRIOR_LAMBDA_EVENT = 0.4
PRIOR_LAMBDA_TEXTURE = 1.0
PRIOR_LAMBDA_PROXY_TEXTURE = 0.8
PRIOR_K_SITE = 8
PRIOR_K_HOUR = 8
PRIOR_K_INTERACTION = 4
PRIOR_LAPLACE_ALPHA = 0.01

# Smoothing / ensemble
SMOOTHING_ALPHA_EVENT = 0.15
SMOOTHING_ALPHA_TEXTURE = 0.35
PERCH_TEMPERATURE = 1.10


# ===========================================================================
# Path resolution
# ===========================================================================


def _find_input(slug: str) -> str | None:
    """Resolve Kaggle input path, supporting both old and new layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
        f"/kaggle/input/notebooks/{KAGGLE_USER}/{slug}",
        f"/kaggle/input/notebooks/{KAGGLE_USER}/{slug}/output",
        f"/kaggle/input/kernels/{KAGGLE_USER}/{slug}",
        f"/kaggle/input/kernels/{KAGGLE_USER}/{slug}/output",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_perch_cache_dir() -> Path | None:
    """Find Perch cache output from Notebook A."""
    path = _find_input(PERCH_CACHE_SLUG)
    if path:
        p = Path(path)
        if (p / "perch_cache.npz").exists():
            return p
        for sub in p.iterdir():
            if sub.is_dir() and (sub / "perch_cache.npz").exists():
                return sub

    # Brute-force walk
    notebooks_dir = Path("/kaggle/input/notebooks")
    if notebooks_dir.exists():
        for root, _dirs, files in os.walk(str(notebooks_dir)):
            if "perch_cache.npz" in files:
                return Path(root)

    # Local testing fallback
    local = Path(os.environ.get("PERCH_CACHE_DIR", "predictions"))
    if (local / "perch_cache.npz").exists():
        return local

    return None


def find_perch_artifacts_dir() -> Path | None:
    """Find Perch ensemble artifacts directory (species mapping + config)."""
    path = _find_input(PERCH_ARTIFACTS_SLUG)
    if path and os.path.exists(path):
        return Path(path)

    local = Path("artifacts")
    if (local / "perch_species_mapping.json").exists():
        return local

    shared = Path(
        "/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
    )
    if shared.exists():
        return shared

    return None


def find_perch_model_dir() -> Path | None:
    """Find Perch v2 TF SavedModel directory."""
    for candidate in PERCH_MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            if (p / "saved_model.pb").exists():
                return p
            for subdir in sorted(p.iterdir()):
                if subdir.is_dir() and (subdir / "saved_model.pb").exists():
                    return subdir
            return p

    # Walk /kaggle/input/ looking for saved_model.pb
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _dirs, files in os.walk(str(kaggle_input)):
            if "saved_model.pb" in files and "perch" in root.lower():
                return Path(root)

    # Local fallback
    for local in [
        Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu"),
        Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2"),
    ]:
        if local.exists():
            return local

    return None


def find_sed_weights_dir() -> Path | None:
    """Find SED model weights directory."""
    path = _find_input(SED_MODEL_SLUG)
    if path and os.path.exists(path):
        return Path(path)

    local = Path("artifacts/weights")
    if (local / "model.onnx").exists():
        return local

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
# Cache loading
# ===========================================================================


def load_perch_cache(cache_dir: Path):
    """Load cached Perch embeddings + mapped scores + metadata from Notebook A."""
    npz = np.load(cache_dir / "perch_cache.npz")
    embeddings = npz["embeddings"]
    mapped_scores = npz["mapped_scores"]
    meta_df = pd.read_parquet(cache_dir / "perch_meta.parquet")

    print(f"  Cache loaded: embeddings {embeddings.shape}, mapped_scores {mapped_scores.shape}")
    print(f"  Metadata: {len(meta_df)} rows, {meta_df['soundscape_id'].nunique()} soundscapes")

    return embeddings, mapped_scores, meta_df


# ===========================================================================
# Perch v2 inference (for labeled soundscapes — probe training)
# ===========================================================================


def load_perch_model(model_dir: str):
    """Load Perch v2 TF SavedModel. Returns (model, infer_fn)."""
    import tensorflow as tf

    print(f"  Loading Perch v2 model from: {model_dir}")
    model = tf.saved_model.load(str(model_dir))
    sigs = list(model.signatures.keys()) if hasattr(model, "signatures") else []

    infer_fn = None
    if hasattr(model, "infer_tf"):
        infer_fn = model.infer_tf
    elif "serving_default" in sigs:
        infer_fn = model.signatures["serving_default"]
    elif callable(model):
        infer_fn = model
    elif sigs:
        infer_fn = model.signatures[sigs[0]]
    else:
        raise RuntimeError("Could not find inference method on Perch model")

    return model, infer_fn


def run_perch_inference(infer_fn, waveform_batch):
    """Run Perch v2 inference. Returns (embeddings, logits)."""
    import tensorflow as tf

    input_tensor = tf.constant(waveform_batch, dtype=tf.float32)
    try:
        output = infer_fn(inputs=input_tensor)
    except TypeError:
        output = infer_fn(input_tensor)

    if isinstance(output, dict):
        embeddings = logits = None
        for key in ["embedding", "embeddings", "output_0"]:
            if key in output:
                embeddings = output[key].numpy()
                break
        for key in ["label", "logits", "logit", "output_1"]:
            if key in output:
                logits = output[key].numpy()
                break
        if embeddings is None or logits is None:
            for _key, val in output.items():
                arr = val.numpy()
                if arr.shape[-1] == 1536 and embeddings is None:
                    embeddings = arr
                elif arr.shape[-1] > 10000 and logits is None:
                    logits = arr
        if embeddings is None or logits is None:
            raise RuntimeError(f"Could not identify outputs from keys: {list(output.keys())}")
    elif isinstance(output, tuple | list):
        arr0 = output[0].numpy() if hasattr(output[0], "numpy") else np.array(output[0])
        arr1 = output[1].numpy() if hasattr(output[1], "numpy") else np.array(output[1])
        if arr0.shape[-1] == 1536:
            embeddings, logits = arr0, arr1
        else:
            logits, embeddings = arr0, arr1
    else:
        raise RuntimeError(f"Unexpected output type: {type(output)}")

    return embeddings.astype(np.float32), logits.astype(np.float32)


def compute_perch_mapped_scores(logits, perch_species_mapping):
    """Map Perch logits to BirdCLEF species scores (234)."""
    birdclef_species = perch_species_mapping["birdclef_species"]
    direct_map = perch_species_mapping["direct_map"]
    genus_proxy_map = perch_species_mapping["genus_proxy_map"]
    n_species = len(birdclef_species)
    n_samples = logits.shape[0]
    n_logits = logits.shape[1]

    sigmoid_logits = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    sigmoid_logits = sigmoid_logits.astype(np.float32)

    scores = np.zeros((n_samples, n_species), dtype=np.float32)
    for i, label in enumerate(birdclef_species):
        if label in direct_map:
            model_idx = direct_map[label] - 1
            if 0 <= model_idx < n_logits:
                scores[:, i] = sigmoid_logits[:, model_idx]
        elif label in genus_proxy_map:
            valid = [idx - 1 for idx in genus_proxy_map[label] if 0 <= idx - 1 < n_logits]
            if valid:
                scores[:, i] = sigmoid_logits[:, valid].max(axis=1)

    return scores


# ===========================================================================
# Probe training (runs inside this notebook on labeled soundscapes)
# ===========================================================================


def _parse_soundscape_metadata(filename):
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


def run_perch_on_labeled_soundscapes(infer_fn, competition_dir, perch_species_mapping):
    """Run Perch v2 on all labeled soundscapes to get training data for probes.

    Returns:
        embeddings: (N, 1536) — raw Perch embeddings
        mapped_scores: (N, 234) — mapped species scores
        meta_rows: list of dicts with filename, window_idx, window_start, window_end, site, hour
    """
    labels_csv = competition_dir / "train_soundscapes_labels.csv"
    train_dir = competition_dir / "train_soundscapes"

    labels_df = pd.read_csv(labels_csv)
    labeled_files = sorted(labels_df["filename"].unique())
    print(f"  Labeled soundscapes: {len(labeled_files)} files")

    all_embeddings = []
    all_mapped_scores = []
    meta_rows = []

    for file_idx, filename in enumerate(labeled_files):
        audio_path = train_dir / filename
        if not audio_path.exists():
            print(f"  WARNING: {audio_path} not found, skipping")
            continue

        # Load audio
        audio, file_sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if file_sr != PERCH_SR:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=PERCH_SR)

        # Ensure exactly 60s
        expected_samples = SOUNDSCAPE_SECONDS * PERCH_SR
        if len(audio) < expected_samples:
            audio = np.pad(audio, (0, expected_samples - len(audio)))
        elif len(audio) > expected_samples:
            audio = audio[:expected_samples]

        # Split into 12 x 5s windows
        windows = []
        for w in range(N_WINDOWS):
            start = w * WINDOW_SAMPLES
            windows.append(audio[start : start + WINDOW_SAMPLES])
        waveform_batch = np.stack(windows, axis=0).astype(np.float32)

        # Run Perch inference
        embeddings, logits = run_perch_inference(infer_fn, waveform_batch)
        mapped_scores = compute_perch_mapped_scores(logits, perch_species_mapping)

        all_embeddings.append(embeddings)
        all_mapped_scores.append(mapped_scores)

        site, hour = _parse_soundscape_metadata(filename)
        for w in range(N_WINDOWS):
            meta_rows.append(
                {
                    "filename": filename,
                    "window_idx": w,
                    "window_start": w * WINDOW_SECONDS,
                    "window_end": (w + 1) * WINDOW_SECONDS,
                    "site": site,
                    "hour": hour,
                }
            )

        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            print(f"    [{file_idx + 1}/{len(labeled_files)}] {filename}")

    all_emb = np.concatenate(all_embeddings, axis=0)
    all_scores = np.concatenate(all_mapped_scores, axis=0)
    print(f"  Labeled embeddings: {all_emb.shape}, mapped_scores: {all_scores.shape}")

    return all_emb, all_scores, meta_rows


def build_labels_matrix(meta_rows, labels_df, birdclef_species):
    """Build binary label matrix (n_windows, n_species) from soundscape labels."""
    n_windows = len(meta_rows)
    n_species = len(birdclef_species)
    species_to_idx = {s: i for i, s in enumerate(birdclef_species)}
    labels_matrix = np.zeros((n_windows, n_species), dtype=np.float32)

    for window_idx, row in enumerate(meta_rows):
        filename = row["filename"]
        start_s = row["window_start"]
        end_s = row["window_end"]

        start_str = (
            f"{int(start_s // 3600):02d}:{int((start_s % 3600) // 60):02d}:{int(start_s % 60):02d}"
        )
        end_str = f"{int(end_s // 3600):02d}:{int((end_s % 3600) // 60):02d}:{int(end_s % 60):02d}"

        mask = (
            (labels_df["filename"] == filename)
            & (labels_df["start"] == start_str)
            & (labels_df["end"] == end_str)
        )
        for _, label_row in labels_df[mask].iterrows():
            label_str = str(label_row["primary_label"])
            if not label_str or label_str == "nan":
                continue
            for label in label_str.split(";"):
                label = label.strip()
                if label in species_to_idx:
                    labels_matrix[window_idx, species_to_idx[label]] = 1.0

    return labels_matrix


def train_probes(pca_embeddings, mapped_scores, labels_matrix, meta_rows, birdclef_species):
    """Train per-species LogReg probes. Returns (probe_models, probe_alphas)."""
    n_positive_per_species = labels_matrix.sum(axis=0)
    probe_models = {}
    probe_alphas = {}
    filenames = np.array([r["filename"] for r in meta_rows])

    for sp_idx, species in enumerate(birdclef_species):
        n_pos = int(n_positive_per_species[sp_idx])
        if n_pos < PROBE_MIN_POSITIVE:
            continue

        # Build features
        features = build_perch_features_for_file_training(
            pca_embeddings, mapped_scores, sp_idx, filenames
        )
        y = labels_matrix[:, sp_idx]

        alpha = PROBE_ALPHA_BASE + PROBE_ALPHA_SCALE * min(
            1.0, (n_pos - PROBE_MIN_POSITIVE) / PROBE_ALPHA_THRESHOLD
        )
        probe_alphas[species] = round(alpha, 4)

        model = LogisticRegression(
            C=PROBE_LOGREG_C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(features, y)
        probe_models[species] = model

    print(f"  Trained {len(probe_models)} probes (skipped {N_SPECIES_2026 - len(probe_models)})")
    return probe_models, probe_alphas


def build_perch_features_for_file_training(pca_embeddings, mapped_scores, species_idx, filenames):
    """Build 37-dim feature vectors for probe training (grouped by file for context)."""
    n_samples = pca_embeddings.shape[0]
    species_scores = mapped_scores[:, species_idx]

    prev_scores = np.zeros(n_samples, dtype=np.float32)
    next_scores = np.zeros(n_samples, dtype=np.float32)
    file_mean_scores = np.zeros(n_samples, dtype=np.float32)
    file_max_scores = np.zeros(n_samples, dtype=np.float32)

    for fname in np.unique(filenames):
        file_indices = np.where(filenames == fname)[0]
        file_scores = species_scores[file_indices]
        file_mean_scores[file_indices] = file_scores.mean()
        file_max_scores[file_indices] = file_scores.max()
        for j, idx in enumerate(file_indices):
            if j > 0:
                prev_scores[idx] = species_scores[file_indices[j - 1]]
            if j < len(file_indices) - 1:
                next_scores[idx] = species_scores[file_indices[j + 1]]

    return np.column_stack(
        [
            pca_embeddings,
            species_scores,
            prev_scores,
            next_scores,
            file_mean_scores,
            file_max_scores,
        ]
    )


def build_metadata_priors(labels_matrix, meta_rows, birdclef_species):
    """Build Bayesian site/hour/interaction prior tables from labeled data."""
    sites = np.array([r["site"] for r in meta_rows])
    hours = np.array([r["hour"] for r in meta_rows])

    unique_sites = sorted({s for s in sites if s is not None})
    unique_hours = sorted({h for h in hours if h is not None})

    global_prior = labels_matrix.mean(axis=0)

    site_priors = {}
    for site in unique_sites:
        mask = sites == site
        n = mask.sum()
        local = labels_matrix[mask].mean(axis=0)
        w = n / (n + PRIOR_K_SITE)
        smoothed = np.maximum(w * local + (1 - w) * global_prior, PRIOR_LAPLACE_ALPHA)
        site_priors[site] = smoothed.tolist()

    hour_priors = {}
    for hour in unique_hours:
        mask = hours == hour
        n = mask.sum()
        local = labels_matrix[mask].mean(axis=0)
        w = n / (n + PRIOR_K_HOUR)
        smoothed = np.maximum(w * local + (1 - w) * global_prior, PRIOR_LAPLACE_ALPHA)
        hour_priors[str(hour)] = smoothed.tolist()

    interaction_priors = {}
    for site in unique_sites:
        for hour in unique_hours:
            mask = (sites == site) & (hours == hour)
            n = mask.sum()
            key = f"{site}_{hour}"
            if n == 0:
                site_p = np.array(site_priors[site])
                hour_p = np.array(hour_priors[str(hour)])
                interaction_priors[key] = np.sqrt(site_p * hour_p).tolist()
            else:
                local = labels_matrix[mask].mean(axis=0)
                site_p = np.array(site_priors[site])
                w = n / (n + PRIOR_K_INTERACTION)
                smoothed = np.maximum(w * local + (1 - w) * site_p, PRIOR_LAPLACE_ALPHA)
                interaction_priors[key] = smoothed.tolist()

    return {
        "global_prior": global_prior.tolist(),
        "site_priors": site_priors,
        "hour_priors": hour_priors,
        "interaction_priors": interaction_priors,
        "unique_sites": unique_sites,
        "unique_hours": [int(h) for h in unique_hours],
    }


# ===========================================================================
# Section 1: SED Inference (EfficientNet-B0 ONNX)
# ===========================================================================


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_full_signal_mel(audio, mel_cfg):
    """Compute mel spectrogram for an entire audio signal."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    norm_type = mel_cfg.get("norm_type", "audioset")
    if norm_type == "amplitude_to_db":
        top_db = mel_cfg.get("top_db", 80.0)
        mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel = np.maximum(mel, mel.max() - top_db)
        mean, std = mel.mean(), max(mel.std(), 1e-6)
        mel = (mel - mean) / std
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
    else:
        mel = np.log(mel + 1e-6)
        mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)
    return mel.astype(np.float32)


def _slice_mel_window(full_mel, start_frame, target_frames):
    """Slice a window from a full-signal mel, padding if needed."""
    total_frames = full_mel.shape[1]
    if start_frame < 0:
        pad_left = -start_frame
        start_frame = 0
    else:
        pad_left = 0
    end_frame = start_frame + target_frames - pad_left
    if end_frame > total_frames:
        end_frame = total_frames
    window = full_mel[:, start_frame:end_frame]
    pad_right = target_frames - pad_left - window.shape[1]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, max(0, pad_right))), mode="constant")
    return window[np.newaxis, np.newaxis, :, :]


def _run_onnx_batch(session, mel_batch, mel_cfg):
    """Run batched ONNX inference."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel_batch})[0]
    if mel_cfg.get("output_type") == "probs":
        return output
    return sigmoid(output)


def _apply_temperature(probs_matrix, temperature):
    """Temperature scaling in logit space."""
    if temperature is None or temperature == 1.0:
        return probs_matrix
    eps = 1e-7
    p = np.clip(probs_matrix, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    return sigmoid(logits / temperature)


def _apply_temporal_smoothing(probs_matrix, taxa_mapping=None):
    """Taxa-aware temporal smoothing."""
    n_segments, n_species = probs_matrix.shape
    if n_segments < 3:
        return probs_matrix
    smoothed = np.copy(probs_matrix)
    if taxa_mapping is None:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for c in range(n_species):
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
        return smoothed

    event_indices = set()
    texture_indices = set()
    for taxa in ["aves", "mammalia", "reptilia"]:
        event_indices.update(taxa_mapping.get(taxa, []))
    for taxa in ["amphibia", "insecta"]:
        texture_indices.update(taxa_mapping.get(taxa, []))

    for c in range(n_species):
        if c in event_indices:
            alpha = 0.15
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha) * probs_matrix[t, c] + alpha * max(prev_val, next_val)
        elif c in texture_indices:
            alpha = 0.35
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha) * probs_matrix[t, c] + alpha * (
                    prev_val + next_val
                ) / 2
        else:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
    return smoothed


def _apply_topn(probs_matrix):
    """TopN: multiply each segment by file-level max per species."""
    return probs_matrix * probs_matrix.max(axis=0, keepdims=True)


def _apply_sed_metadata_priors(probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map):
    """Apply site/hour metadata priors for SED."""
    if metadata_priors is None:
        return probs_matrix
    n_segments, n_species = probs_matrix.shape
    lambda_map = {"aves": 0.3, "amphibia": 0.5, "insecta": 0.5, "mammalia": 0.2, "reptilia": 0.2}

    idx_to_taxa = {}
    if taxa_mapping:
        for taxa_class, indices in taxa_mapping.items():
            for idx in indices:
                idx_to_taxa[idx] = taxa_class

    site_hour_priors = metadata_priors.get("site_hour_species", {})
    site_priors = metadata_priors.get("site_species", {})
    hour_priors = metadata_priors.get("hour_species", {})
    global_priors = metadata_priors.get("global_species", {})

    site_str = site if site else None
    hour_str = str(hour).zfill(2) if hour is not None else None

    prior_vector = np.ones(n_species, dtype=np.float32)
    for idx_str, species_code in idx_map.items():
        idx = int(idx_str)
        if idx >= n_species:
            continue
        prior_val = None
        if site_str and hour_str:
            prior_val = site_hour_priors.get(site_str, {}).get(hour_str, {}).get(species_code)
        if prior_val is None and site_str:
            prior_val = site_priors.get(site_str, {}).get(species_code)
        if prior_val is None and hour_str:
            prior_val = hour_priors.get(hour_str, {}).get(species_code)
        if prior_val is None:
            prior_val = global_priors.get(species_code)
        if prior_val is None:
            continue
        taxa = idx_to_taxa.get(idx, "aves")
        lam = lambda_map.get(taxa, 0.3)
        prior_vector[idx] = prior_val**lam

    return probs_matrix * prior_vector[np.newaxis, :]


def run_sed_inference(
    session, audio_path, mel_cfg, species_mapping, taxa_mapping=None, sed_metadata_priors=None
):
    """Process one soundscape using full-signal mel."""
    sr = mel_cfg["sr"]
    hop_length = mel_cfg["hop_length"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2
    target_frames = mel_cfg.get("target_frames", 1000)

    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    total_duration = len(audio) / sr
    context_samples = int(context_pad * sr)
    tta_shift_samples = int(2.5 * sr)
    pad_samples = context_samples + tta_shift_samples
    audio_padded = np.pad(audio, (pad_samples, pad_samples), mode="constant")
    full_mel = compute_full_signal_mel(audio_padded, mel_cfg)
    origin_frame = int(pad_samples / hop_length)

    segment_times = []
    segment_start_s = 0.0
    all_slices = []
    segment_idx = 0

    while segment_start_s < total_duration:
        window_center_s = segment_start_s - context_pad
        if ENABLE_TTA:
            shifts = [
                (window_center_s, 0.50),
                (window_center_s - 2.5, 0.25),
                (window_center_s + 2.5, 0.25),
            ]
        else:
            shifts = [(window_center_s, 1.0)]
        for shift_start_s, weight in shifts:
            start_frame = origin_frame + int(shift_start_s * sr / hop_length)
            mel_slice = _slice_mel_window(full_mel, start_frame, target_frames)
            all_slices.append((segment_idx, weight, mel_slice))
        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration
        segment_idx += 1

    n_segments = segment_idx
    mel_batch = np.concatenate([s[2] for s in all_slices], axis=0)
    all_probs = _run_onnx_batch(session, mel_batch, mel_cfg)

    probs_matrix = np.zeros((n_segments, all_probs.shape[1]), dtype=np.float32)
    for i, (seg_idx, weight, _) in enumerate(all_slices):
        probs_matrix[seg_idx] += weight * all_probs[i]

    temperature = mel_cfg.get("temperature", 1.0)
    if temperature != 1.0:
        probs_matrix = _apply_temperature(probs_matrix, temperature)
    if ENABLE_SED_TEMPORAL_SMOOTHING:
        probs_matrix = _apply_temporal_smoothing(probs_matrix, taxa_mapping)
    if ENABLE_SED_TOPN:
        probs_matrix = _apply_topn(probs_matrix)

    site, hour = _parse_soundscape_metadata(audio_path)
    if sed_metadata_priors is not None and (site is not None or hour is not None):
        idx_map = (
            species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
        )
        probs_matrix = _apply_sed_metadata_priors(
            probs_matrix, sed_metadata_priors, site, hour, taxa_mapping, idx_map
        )

    return probs_matrix, segment_times


def map_sed_to_species(probs_matrix, species_mapping):
    """Map SED outputs to 234 BirdCLEF species."""
    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    n_segments = probs_matrix.shape[0]
    mapped = np.zeros((n_segments, len(species_2026)), dtype=np.float32)
    species_to_col = {sp: i for i, sp in enumerate(species_2026)}
    for idx_str, name_2026 in idx_map.items():
        col_idx = species_to_col.get(name_2026)
        if col_idx is not None:
            mapped[:, col_idx] = probs_matrix[:, int(idx_str)]
    return mapped


# ===========================================================================
# Section 2: Perch Post-Processing (applied to cached test embeddings)
# ===========================================================================


def build_perch_features_for_file(pca_embeddings, mapped_scores, species_idx):
    """Build 37-dim feature vector for inference (single file, all windows)."""
    n_windows = pca_embeddings.shape[0]
    species_scores = mapped_scores[:, species_idx]
    prev_scores = np.zeros(n_windows, dtype=np.float32)
    next_scores = np.zeros(n_windows, dtype=np.float32)
    if n_windows > 1:
        prev_scores[1:] = species_scores[:-1]
        next_scores[:-1] = species_scores[1:]
    file_mean = species_scores.mean()
    file_max = species_scores.max()
    return np.column_stack(
        [
            pca_embeddings,
            species_scores,
            prev_scores,
            next_scores,
            np.full(n_windows, file_mean, dtype=np.float32),
            np.full(n_windows, file_max, dtype=np.float32),
        ]
    )


def apply_perch_probes(pca_embeddings, mapped_scores, probe_models, probe_alphas, birdclef_species):
    """Apply per-species LogReg probes to refine Perch scores."""
    perch_probs = np.copy(mapped_scores)
    n_probed = 0
    for sp_idx, species in enumerate(birdclef_species):
        if species not in probe_models:
            continue
        probe = probe_models[species]
        alpha = probe_alphas.get(species, 0.25)
        features = build_perch_features_for_file(pca_embeddings, mapped_scores, sp_idx)
        try:
            probe_score = probe.predict_proba(features)[:, 1]
        except Exception:
            continue
        base_score = mapped_scores[:, sp_idx]
        perch_probs[:, sp_idx] = (1 - alpha) * base_score + alpha * probe_score
        n_probed += 1
    return perch_probs, n_probed


def apply_perch_metadata_priors(
    perch_probs, metadata_priors, site, hour, birdclef_species, taxa_classes, unmapped_set
):
    """Apply Bayesian metadata priors to Perch predictions."""
    if metadata_priors is None:
        return perch_probs

    n_segments, n_species = perch_probs.shape
    genus_proxy_set = set()  # TODO: load from species mapping if needed

    global_prior = np.array(metadata_priors.get("global_prior", []), dtype=np.float32)
    site_priors = metadata_priors.get("site_priors", {})
    hour_priors = metadata_priors.get("hour_priors", {})
    interaction_priors = metadata_priors.get("interaction_priors", {})

    if len(global_prior) == 0:
        return perch_probs

    site_str = site if site else None
    hour_str = str(hour) if hour is not None else None
    interaction_key = f"{site_str}_{hour_str}" if site_str and hour_str else None

    prior_vector = global_prior.copy()
    if interaction_key and interaction_key in interaction_priors:
        prior_vector = np.array(interaction_priors[interaction_key], dtype=np.float32)
    elif site_str and site_str in site_priors:
        prior_vector = np.array(site_priors[site_str], dtype=np.float32)
    elif hour_str and hour_str in hour_priors:
        prior_vector = np.array(hour_priors[hour_str], dtype=np.float32)

    if len(prior_vector) != n_species:
        return perch_probs

    lambda_vector = np.full(n_species, PRIOR_LAMBDA_EVENT, dtype=np.float32)
    for sp_idx, species in enumerate(birdclef_species):
        if species in unmapped_set:
            lambda_vector[sp_idx] = 0.0
            continue
        taxa = taxa_classes.get(species, "Aves")
        if species in genus_proxy_set:
            lambda_vector[sp_idx] = PRIOR_LAMBDA_PROXY_TEXTURE
        elif taxa in ("Amphibia", "Insecta"):
            lambda_vector[sp_idx] = PRIOR_LAMBDA_TEXTURE
        elif taxa in ("Mammalia", "Reptilia"):
            lambda_vector[sp_idx] = PRIOR_LAMBDA_EVENT * 0.5

    prior_multiplier = np.power(np.maximum(prior_vector, 1e-10), lambda_vector)
    return perch_probs * prior_multiplier[np.newaxis, :]


def apply_perch_temporal_smoothing(
    perch_probs, taxa_classes, birdclef_species, alpha_event=0.15, alpha_texture=0.35
):
    """Taxa-aware temporal smoothing for Perch predictions."""
    n_segments, n_species = perch_probs.shape
    if n_segments < 3:
        return perch_probs
    smoothed = np.copy(perch_probs)
    event_classes = {"Aves", "Mammalia", "Reptilia"}
    texture_classes = {"Amphibia", "Insecta"}

    for sp_idx, species in enumerate(birdclef_species):
        taxa = taxa_classes.get(species, "Aves")
        if taxa in event_classes:
            alpha = alpha_event
            for t in range(n_segments):
                prev_val = perch_probs[max(0, t - 1), sp_idx]
                next_val = perch_probs[min(n_segments - 1, t + 1), sp_idx]
                smoothed[t, sp_idx] = (1 - alpha) * perch_probs[t, sp_idx] + alpha * max(
                    prev_val, next_val
                )
        elif taxa in texture_classes:
            alpha = alpha_texture
            for t in range(n_segments):
                prev_val = perch_probs[max(0, t - 1), sp_idx]
                next_val = perch_probs[min(n_segments - 1, t + 1), sp_idx]
                smoothed[t, sp_idx] = (1 - alpha) * perch_probs[t, sp_idx] + alpha * (
                    prev_val + next_val
                ) / 2
        else:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smoothed[:, sp_idx] = np.convolve(perch_probs[:, sp_idx], kernel, mode="same")
    return smoothed


# ===========================================================================
# Section 3: Ensemble Blending
# ===========================================================================


def blend_ensemble(
    sed_probs_mapped, perch_probs, unmapped_indices, sed_weight=0.3, perch_weight=0.7
):
    """Blend SED and Perch predictions."""
    final_probs = sed_weight * sed_probs_mapped + perch_weight * perch_probs
    for idx in unmapped_indices:
        final_probs[:, idx] = sed_probs_mapped[:, idx]
    return final_probs


# ===========================================================================
# Section 4: Main
# ===========================================================================


def main():
    print("=" * 70)
    print("BirdCLEF 2026 Scoring Kernel (Notebook B)")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------------
    competition_dir = find_competition_dir()
    test_dir = competition_dir / "test_soundscapes"
    sample_sub_path = competition_dir / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")

    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not sample_sub_path.exists():
        sample_sub_path = Path(os.environ.get("SAMPLE_SUBMISSION", "sample_submission.csv"))
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    perch_cache_dir = find_perch_cache_dir()
    perch_cache_available = perch_cache_dir is not None
    perch_artifacts_dir = find_perch_artifacts_dir()
    perch_model_dir = find_perch_model_dir()

    # Need cache + artifacts + Perch model for probe training
    perch_available = (
        perch_cache_available and perch_artifacts_dir is not None and perch_model_dir is not None
    )

    if PERCH_ONLY:
        sed_weights_dir = None
        sed_available = False
    else:
        sed_weights_dir = find_sed_weights_dir()
        sed_available = sed_weights_dir is not None and (sed_weights_dir / "model.onnx").exists()

    print(f"\nCompetition data: {competition_dir}")
    print(f"Perch cache: {perch_cache_dir} (available={perch_cache_available})")
    print(f"Perch artifacts: {perch_artifacts_dir}")
    print(f"Perch model: {perch_model_dir}")
    print(f"SED weights: {sed_weights_dir} (available={sed_available})")

    if not perch_available and not sed_available:
        raise FileNotFoundError("Neither Perch nor SED available!")

    # -------------------------------------------------------------------
    # Load species mapping (JSON, no pickle — always safe)
    # -------------------------------------------------------------------
    perch_species_mapping = None
    perch_birdclef_species = None
    perch_taxa_classes = None
    unmapped_set = set()
    unmapped_indices = set()

    if perch_available:
        for name in ["perch_species_mapping.json", "species_mapping.json"]:
            p = perch_artifacts_dir / name
            if p.exists():
                with open(p) as f:
                    perch_species_mapping = json.load(f)
                break

        if perch_species_mapping:
            perch_birdclef_species = perch_species_mapping["birdclef_species"]
            perch_taxa_classes = perch_species_mapping.get("taxa_classes", {})
            unmapped_set = set(perch_species_mapping.get("unmapped", []))
            species_to_idx = {sp: i for i, sp in enumerate(perch_birdclef_species)}
            unmapped_indices = {species_to_idx[sp] for sp in unmapped_set if sp in species_to_idx}
            print(f"  Species: {len(perch_birdclef_species)}, unmapped: {len(unmapped_set)}")
        else:
            print("  WARNING: Species mapping not found")
            perch_available = False

    # -------------------------------------------------------------------
    # Load Perch model + run on labeled soundscapes + train probes
    # -------------------------------------------------------------------
    pca_transformer = None
    probe_models = {}
    probe_alphas = {}
    perch_metadata_priors = None
    embeddings_all = None
    mapped_scores_all = None
    meta_df = None

    if perch_available:
        print("\n--- Loading Perch Model ---")
        t_perch = time.time()
        _, infer_fn = load_perch_model(str(perch_model_dir))

        # Probe with dummy
        dummy = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
        test_emb, test_logits = run_perch_inference(infer_fn, dummy)
        print(f"  Perch shapes: embeddings={test_emb.shape}, logits={test_logits.shape}")
        print(f"  Model loaded in {time.time() - t_perch:.1f}s")

        # Run Perch on labeled soundscapes
        print("\n--- Running Perch on Labeled Soundscapes ---")
        t_labeled = time.time()
        labeled_emb, labeled_scores, labeled_meta = run_perch_on_labeled_soundscapes(
            infer_fn, competition_dir, perch_species_mapping
        )
        print(f"  Labeled inference: {time.time() - t_labeled:.1f}s")

        # Build labels
        print("\n--- Training Probes ---")
        t_train = time.time()
        labels_csv = competition_dir / "train_soundscapes_labels.csv"
        labels_df = pd.read_csv(labels_csv)
        labels_matrix = build_labels_matrix(labeled_meta, labels_df, perch_birdclef_species)
        n_pos_species = (labels_matrix.sum(axis=0) > 0).sum()
        print(f"  Labels: {labels_matrix.sum():.0f} positives across {n_pos_species} species")

        # PCA
        pca_transformer = PCA(n_components=PROBE_PCA_COMPONENTS, random_state=42)
        labeled_pca = pca_transformer.fit_transform(labeled_emb)
        print(
            f"  PCA: {labeled_emb.shape[1]} → {PROBE_PCA_COMPONENTS} dims "
            f"({pca_transformer.explained_variance_ratio_.sum():.3f} variance)"
        )

        # Train probes
        probe_models, probe_alphas = train_probes(
            labeled_pca, labeled_scores, labels_matrix, labeled_meta, perch_birdclef_species
        )

        # Build metadata priors
        perch_metadata_priors = build_metadata_priors(
            labels_matrix, labeled_meta, perch_birdclef_species
        )
        print(
            f"  Priors: {len(perch_metadata_priors['unique_sites'])} sites, "
            f"{len(perch_metadata_priors['unique_hours'])} hours"
        )
        print(f"  Total training time: {time.time() - t_train:.1f}s")

        # Load test cache
        print("\n--- Loading Test Cache ---")
        embeddings_all, mapped_scores_all, meta_df = load_perch_cache(perch_cache_dir)

    # -------------------------------------------------------------------
    # Load SED artifacts (Phase C only)
    # -------------------------------------------------------------------
    onnx_session = None
    sed_mel_cfg = None
    sed_species_mapping = None
    sed_taxa_mapping = None
    sed_metadata_priors = None

    if sed_available:
        print("\n--- Loading SED Model ---")
        onnx_path = sed_weights_dir / "model.onnx"
        with open(sed_weights_dir / "mel_config.json") as f:
            sed_mel_cfg = json.load(f)
        with open(sed_weights_dir / "species_mapping.json") as f:
            sed_species_mapping = json.load(f)

        taxa_path = sed_weights_dir / "taxa_mapping.json"
        if taxa_path.exists():
            with open(taxa_path) as f:
                sed_taxa_mapping = json.load(f)
        priors_path = sed_weights_dir / "metadata_priors.json"
        if priors_path.exists():
            with open(priors_path) as f:
                sed_metadata_priors = json.load(f)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = 1
        onnx_session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

    # -------------------------------------------------------------------
    # Determine mode
    # -------------------------------------------------------------------
    if sed_available and perch_available:
        mode = "ensemble"
        sed_weight, perch_weight = 0.3, 0.7
    elif perch_available:
        mode = "perch_only"
        sed_weight, perch_weight = 0.0, 1.0
    elif sed_available:
        mode = "sed_only"
        sed_weight, perch_weight = 1.0, 0.0
    else:
        raise RuntimeError("No models available")

    print(f"\n--- Mode: {mode.upper()} ---")

    # -------------------------------------------------------------------
    # Determine species list and soundscape ordering
    # -------------------------------------------------------------------
    if perch_available and perch_birdclef_species:
        species_list_2026 = perch_birdclef_species
    elif sed_available:
        species_list_2026 = sed_species_mapping["birdclef2026_species"]
    else:
        species_list_2026 = []

    if perch_available and meta_df is not None:
        soundscape_order = list(dict.fromkeys(meta_df["soundscape_id"]))
        print(f"Soundscapes in cache: {len(soundscape_order)}")
    else:
        soundscape_order = []

    if mode == "sed_only":
        audio_files = sorted(test_dir.glob("*.ogg"))
        if not audio_files:
            train_dir = competition_dir / "train_soundscapes"
            if train_dir.exists():
                audio_files = sorted(train_dir.glob("*.ogg"))[:3]
        soundscape_order = [f.stem for f in audio_files]
        soundscape_to_path = {f.stem: f for f in audio_files}
    else:
        soundscape_to_path = {}
        if sed_available:
            audio_files = sorted(test_dir.glob("*.ogg"))
            if not audio_files:
                train_dir = competition_dir / "train_soundscapes"
                if train_dir.exists():
                    audio_files = sorted(train_dir.glob("*.ogg"))
            soundscape_to_path = {f.stem: f for f in audio_files}

    # -------------------------------------------------------------------
    # Process each soundscape
    # -------------------------------------------------------------------
    all_predictions = []
    timing_stats = {"perch_pp": [], "sed": [], "total": []}

    for sc_idx, soundscape_id in enumerate(soundscape_order):
        t_file_start = time.time()
        perch_probs = None
        segment_times = None

        # ----- a. Perch post-processing (from cache) -----
        if perch_available and meta_df is not None:
            t_pp = time.time()
            file_mask = meta_df["soundscape_id"] == soundscape_id
            file_rows = meta_df[file_mask].sort_values("window_idx")
            file_indices = file_rows.index.values

            file_embeddings = embeddings_all[file_indices]
            file_mapped_scores = mapped_scores_all[file_indices]
            site = file_rows.iloc[0]["site"]
            hour = file_rows.iloc[0]["hour"]
            segment_times = file_rows["end_time"].tolist()

            pca_embeddings = pca_transformer.transform(file_embeddings)

            perch_probs, _ = apply_perch_probes(
                pca_embeddings,
                file_mapped_scores,
                probe_models,
                probe_alphas,
                perch_birdclef_species,
            )

            if perch_metadata_priors is not None and (site is not None or hour is not None):
                perch_probs = apply_perch_metadata_priors(
                    perch_probs,
                    perch_metadata_priors,
                    site,
                    hour,
                    perch_birdclef_species,
                    perch_taxa_classes,
                    unmapped_set,
                )

            perch_probs = apply_perch_temporal_smoothing(
                perch_probs,
                perch_taxa_classes,
                perch_birdclef_species,
                alpha_event=SMOOTHING_ALPHA_EVENT,
                alpha_texture=SMOOTHING_ALPHA_TEXTURE,
            )

            if PERCH_TEMPERATURE != 1.0:
                perch_probs = _apply_temperature(perch_probs, PERCH_TEMPERATURE)

            timing_stats["perch_pp"].append(time.time() - t_pp)

        # ----- b. SED inference -----
        sed_probs_mapped = None
        if sed_available and soundscape_id in soundscape_to_path:
            t_sed = time.time()
            sed_probs_raw, sed_segment_times = run_sed_inference(
                onnx_session,
                str(soundscape_to_path[soundscape_id]),
                sed_mel_cfg,
                sed_species_mapping,
                taxa_mapping=sed_taxa_mapping,
                sed_metadata_priors=sed_metadata_priors,
            )
            sed_probs_mapped = map_sed_to_species(sed_probs_raw, sed_species_mapping)
            if segment_times is None:
                segment_times = sed_segment_times
            timing_stats["sed"].append(time.time() - t_sed)

        # ----- c. Blend -----
        if segment_times is None:
            continue

        if mode == "ensemble" and sed_probs_mapped is not None and perch_probs is not None:
            min_seg = min(sed_probs_mapped.shape[0], perch_probs.shape[0])
            final_probs = blend_ensemble(
                sed_probs_mapped[:min_seg],
                perch_probs[:min_seg],
                unmapped_indices,
                sed_weight=sed_weight,
                perch_weight=perch_weight,
            )
            segment_times = segment_times[:min_seg]
        elif perch_probs is not None:
            final_probs = perch_probs
        elif sed_probs_mapped is not None:
            final_probs = sed_probs_mapped
        else:
            continue

        # ----- d. Build rows -----
        for seg_idx, end_time in enumerate(segment_times):
            if seg_idx >= final_probs.shape[0]:
                break
            row_id = f"{soundscape_id}_{end_time}"
            row = {"row_id": row_id}
            for sp_idx, species in enumerate(species_list_2026):
                row[species] = float(final_probs[seg_idx, sp_idx])
            all_predictions.append(row)

        t_elapsed = time.time() - t_file_start
        timing_stats["total"].append(t_elapsed)

        if (sc_idx + 1) % 50 == 0 or sc_idx == 0 or (sc_idx + 1) == len(soundscape_order):
            print(f"  [{sc_idx + 1}/{len(soundscape_order)}] {soundscape_id}: ({t_elapsed:.2f}s)")

    print(f"\nTotal predictions: {len(all_predictions)} rows")

    # -------------------------------------------------------------------
    # Timing summary
    # -------------------------------------------------------------------
    print("\n--- Timing Summary ---")
    for component, times in timing_stats.items():
        if times:
            print(f"  {component}: mean={np.mean(times):.3f}s, total={np.sum(times):.1f}s")

    # -------------------------------------------------------------------
    # Build submission CSV
    # -------------------------------------------------------------------
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed_total = time.time() - START_TIME
    print(f"\nSubmission saved to {output_path} ({len(submission)} rows)")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"Mode: {mode}")


if __name__ == "__main__":
    main()
