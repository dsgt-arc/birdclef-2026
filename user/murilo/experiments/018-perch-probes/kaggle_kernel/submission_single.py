#!/usr/bin/env python3
"""BirdCLEF 2026 Single-Kernel Perch Scoring Notebook.

Single-kernel architecture matching reference notebooks (chiranjithdharma, yuriygreben):
  1. Load pre-extracted labeled embeddings from Kaggle dataset (~1s)
  2. Train PCA + LogReg probes + metadata priors from scratch (~2s)
  3. Run Perch v2 on all test soundscapes with ThreadPoolExecutor (~80 min)
  4. Apply probes + priors + smoothing to test embeddings (~5s)
  5. Submit

This avoids the two-notebook chain (kernel_sources) which failed because Kaggle uses
cached push-run outputs (train files) instead of re-running with test data.

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
from concurrent.futures import ThreadPoolExecutor

START_TIME = time.time()
TIMEOUT_SECONDS = 88 * 60  # 88 min safety cutoff

KAGGLE_USER = "murilogustineli"
PERCH_ONLY = True


def _find_input_early(slug: str) -> str:
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")
print(f"PERCH_ONLY mode: {PERCH_ONLY}")

# Install TF 2.20
TF_WHEEL_CANDIDATES = [
    _find_input_early("birdclef-tf220-wheel"),
    "/kaggle/input/birdclef-tf220-wheel",
    f"/kaggle/input/datasets/{KAGGLE_USER}/birdclef-tf220-wheel",
]
for tf_wheel_dir in TF_WHEEL_CANDIDATES:
    if os.path.exists(tf_wheel_dir):
        print(f"Installing TF 2.20 from {tf_wheel_dir}")
        for whl in sorted(os.listdir(tf_wheel_dir)):
            if whl.endswith(".whl"):
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        os.path.join(tf_wheel_dir, whl),
                    ]
                )
        break
else:
    print("WARNING: TF 2.20 wheel not found")

# Install onnxruntime for SED (Phase C only)
if not PERCH_ONLY:
    DEPS_DIR = _find_input_early("birdclef-deps")
    if os.path.exists(DEPS_DIR):
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
# Imports
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
    pass  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERCH_SR = 32000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SECONDS
SOUNDSCAPE_SECONDS = 60
N_WINDOWS = SOUNDSCAPE_SECONDS // WINDOW_SECONDS
N_SPECIES_2026 = 234

# Kaggle dataset slugs
LABELED_EMB_SLUG = "birdclef-2026-perch-labeled-embeddings"
PERCH_ARTIFACTS_SLUG = "birdclef-2026-perch-ensemble-v1"
SED_MODEL_SLUG = "birdclef-2026-effnet-b0-sed-v4b-weights"

# Perch model candidates
PERCH_MODEL_CANDIDATES = [
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier",
]

# Probe training hyperparameters
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
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
        f"/kaggle/input/notebooks/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_labeled_embeddings_dir() -> Path | None:
    """Find pre-extracted labeled embeddings dataset."""
    path = _find_input(LABELED_EMB_SLUG)
    if path:
        p = Path(path)
        if (p / "labeled_perch_arrays.npz").exists():
            return p

    # Local fallback
    local = Path(
        "/storage/project/ps-clef2026_birdclef-0/shared/datasets/birdclef-2026/perch-labeled-embeddings"
    )
    if local.exists() and (local / "labeled_perch_arrays.npz").exists():
        return local

    return None


def find_perch_artifacts_dir() -> Path | None:
    """Find species mapping + config JSON."""
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
    """Find Perch v2 TF SavedModel."""
    for candidate in PERCH_MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            if (p / "saved_model.pb").exists():
                return p
            for subdir in sorted(p.iterdir()):
                if subdir.is_dir() and (subdir / "saved_model.pb").exists():
                    return subdir
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _dirs, files in os.walk(str(kaggle_input)):
            if "saved_model.pb" in files and "perch" in root.lower():
                return Path(root)

    for local in [
        Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu"),
        Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2"),
    ]:
        if local.exists():
            return local
    return None


def find_sed_weights_dir() -> Path | None:
    path = _find_input(SED_MODEL_SLUG)
    if path and os.path.exists(path):
        return Path(path)
    local = Path("artifacts/weights")
    if (local / "model.onnx").exists():
        return local
    return None


def find_competition_dir() -> Path:
    for c in ["/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"]:
        if os.path.exists(c):
            return Path(c)
    return Path(os.environ.get("COMPETITION_DIR", "data/birdclef-2026"))


# ===========================================================================
# Perch v2 inference
# ===========================================================================


def load_perch_model(model_dir: str):
    """Load Perch v2 TF SavedModel. Returns (model, infer_fn)."""
    import tensorflow as tf

    print(f"  Loading Perch v2 from: {model_dir}")
    model = tf.saved_model.load(str(model_dir))
    sigs = list(model.signatures.keys()) if hasattr(model, "signatures") else []

    if hasattr(model, "infer_tf"):
        return model, model.infer_tf
    if "serving_default" in sigs:
        return model, model.signatures["serving_default"]
    if callable(model):
        return model, model
    if sigs:
        return model, model.signatures[sigs[0]]
    raise RuntimeError("Could not find Perch inference method")


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
            raise RuntimeError(f"Could not identify outputs: {list(output.keys())}")
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
    """Map Perch logits to 234 BirdCLEF species scores (with genus proxies)."""
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


def load_and_window_audio(audio_path, sr=PERCH_SR):
    """Load 60s soundscape → (12, 160000) float32."""
    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    expected = SOUNDSCAPE_SECONDS * sr
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    elif len(audio) > expected:
        audio = audio[:expected]

    windows = [audio[w * WINDOW_SAMPLES : (w + 1) * WINDOW_SAMPLES] for w in range(N_WINDOWS)]
    return np.stack(windows, axis=0).astype(np.float32)


def _parse_soundscape_metadata(filename):
    """Parse site and hour from filename."""
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


# ===========================================================================
# Probe training (on pre-extracted labeled embeddings)
# ===========================================================================


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


def build_perch_features_for_file_training(pca_embeddings, mapped_scores, species_idx, filenames):
    """Build 37-dim feature vectors for probe training (grouped by file)."""
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


def train_probes(pca_embeddings, mapped_scores, labels_matrix, filenames, birdclef_species):
    """Train per-species LogReg probes. Returns (probe_models, probe_alphas)."""
    n_positive_per_species = labels_matrix.sum(axis=0)
    probe_models = {}
    probe_alphas = {}

    for sp_idx, species in enumerate(birdclef_species):
        n_pos = int(n_positive_per_species[sp_idx])
        if n_pos < PROBE_MIN_POSITIVE:
            continue

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


def build_metadata_priors(labels_matrix, meta_rows, birdclef_species):
    """Build Bayesian site/hour/interaction prior tables."""
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
# Perch post-processing (applied to test embeddings)
# ===========================================================================


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _apply_temperature(probs_matrix, temperature):
    if temperature is None or temperature == 1.0:
        return probs_matrix
    eps = 1e-7
    p = np.clip(probs_matrix, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    return sigmoid(logits / temperature)


def build_perch_features_for_file(pca_embeddings, mapped_scores, species_idx):
    """Build 37-dim features for inference (single file, all windows)."""
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
        perch_probs[:, sp_idx] = (1 - alpha) * mapped_scores[:, sp_idx] + alpha * probe_score
        n_probed += 1
    return perch_probs, n_probed


def apply_perch_metadata_priors(
    perch_probs, metadata_priors, site, hour, birdclef_species, taxa_classes, unmapped_set
):
    """Apply Bayesian metadata priors to Perch predictions."""
    if metadata_priors is None:
        return perch_probs

    n_segments, n_species = perch_probs.shape
    global_prior = np.array(metadata_priors.get("global_prior", []), dtype=np.float32)
    if len(global_prior) == 0:
        return perch_probs

    site_priors = metadata_priors.get("site_priors", {})
    hour_priors = metadata_priors.get("hour_priors", {})
    interaction_priors = metadata_priors.get("interaction_priors", {})

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
        elif taxa_classes.get(species, "Aves") in ("Amphibia", "Insecta"):
            lambda_vector[sp_idx] = PRIOR_LAMBDA_TEXTURE
        elif taxa_classes.get(species, "Aves") in ("Mammalia", "Reptilia"):
            lambda_vector[sp_idx] = PRIOR_LAMBDA_EVENT * 0.5

    prior_multiplier = np.power(np.maximum(prior_vector, 1e-10), lambda_vector)
    return perch_probs * prior_multiplier[np.newaxis, :]


def apply_perch_temporal_smoothing(
    perch_probs, taxa_classes, birdclef_species, alpha_event=0.15, alpha_texture=0.35
):
    """Taxa-aware temporal smoothing."""
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
# SED Inference (Phase C only — kept for future ensemble)
# ===========================================================================

# TODO: Add SED inference functions from submission_scoring.py when PERCH_ONLY=False


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 70)
    print("BirdCLEF 2026 Single-Kernel Perch Scoring")
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

    labeled_emb_dir = find_labeled_embeddings_dir()
    perch_artifacts_dir = find_perch_artifacts_dir()
    perch_model_dir = find_perch_model_dir()

    print(f"\nCompetition data: {competition_dir}")
    print(f"Labeled embeddings: {labeled_emb_dir}")
    print(f"Perch artifacts: {perch_artifacts_dir}")
    print(f"Perch model: {perch_model_dir}")

    if labeled_emb_dir is None:
        raise FileNotFoundError("Labeled embeddings dataset not found!")
    if perch_artifacts_dir is None:
        raise FileNotFoundError("Perch artifacts (species mapping) not found!")
    if perch_model_dir is None:
        raise FileNotFoundError("Perch v2 model not found!")

    # -------------------------------------------------------------------
    # Load species mapping
    # -------------------------------------------------------------------
    perch_species_mapping = None
    for name in ["perch_species_mapping.json", "species_mapping.json"]:
        p = perch_artifacts_dir / name
        if p.exists():
            with open(p) as f:
                perch_species_mapping = json.load(f)
            break

    if perch_species_mapping is None:
        raise FileNotFoundError("Species mapping not found!")

    birdclef_species = perch_species_mapping["birdclef_species"]
    taxa_classes = perch_species_mapping.get("taxa_classes", {})
    unmapped_set = set(perch_species_mapping.get("unmapped", []))
    print(f"  Species: {len(birdclef_species)}, unmapped: {len(unmapped_set)}")

    # -------------------------------------------------------------------
    # Phase 1: Load labeled embeddings + train probes
    # -------------------------------------------------------------------
    print("\n--- Phase 1: Load Labeled Embeddings + Train Probes ---")
    t_phase1 = time.time()

    # Load pre-extracted labeled embeddings
    npz = np.load(labeled_emb_dir / "labeled_perch_arrays.npz")
    labeled_embeddings = npz["embeddings"]  # (792, 1536)
    labeled_mapped_scores = npz["mapped_scores"]  # (792, 234)
    labeled_meta_df = pd.read_parquet(labeled_emb_dir / "labeled_perch_meta.parquet")

    print(f"  Labeled: embeddings {labeled_embeddings.shape}, scores {labeled_mapped_scores.shape}")
    print(f"  Files: {labeled_meta_df['filename'].nunique()}")

    # Build meta_rows list for label matrix construction
    meta_rows = []
    for _, row in labeled_meta_df.iterrows():
        meta_rows.append(
            {
                "filename": row["filename"],
                "window_idx": int(row["window_idx"]),
                "window_start": float(row["window_start"]),
                "window_end": float(row["window_end"]),
                "site": row.get("site"),
                "hour": int(row["hour"]) if pd.notna(row.get("hour")) else None,
            }
        )

    # Build labels
    labels_csv = competition_dir / "train_soundscapes_labels.csv"
    labels_df = pd.read_csv(labels_csv)
    labels_matrix = build_labels_matrix(meta_rows, labels_df, birdclef_species)
    n_pos_species = (labels_matrix.sum(axis=0) > 0).sum()
    print(f"  Labels: {labels_matrix.sum():.0f} positives across {n_pos_species} species")

    # PCA
    pca_transformer = PCA(n_components=PROBE_PCA_COMPONENTS, random_state=42)
    labeled_pca = pca_transformer.fit_transform(labeled_embeddings)
    print(
        f"  PCA: {labeled_embeddings.shape[1]} → {PROBE_PCA_COMPONENTS} dims "
        f"({pca_transformer.explained_variance_ratio_.sum():.3f} variance)"
    )

    # Train probes
    filenames_arr = np.array([r["filename"] for r in meta_rows])
    probe_models, probe_alphas = train_probes(
        labeled_pca, labeled_mapped_scores, labels_matrix, filenames_arr, birdclef_species
    )

    # Build metadata priors
    metadata_priors = build_metadata_priors(labels_matrix, meta_rows, birdclef_species)
    print(
        f"  Priors: {len(metadata_priors['unique_sites'])} sites, "
        f"{len(metadata_priors['unique_hours'])} hours"
    )
    print(f"  Phase 1 time: {time.time() - t_phase1:.1f}s")

    # -------------------------------------------------------------------
    # Phase 2: Load Perch model + run on test soundscapes
    # -------------------------------------------------------------------
    print("\n--- Phase 2: Perch Inference on Test Soundscapes ---")
    t_phase2 = time.time()

    _, infer_fn = load_perch_model(str(perch_model_dir))

    # Probe with dummy
    dummy = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    test_emb, test_logits = run_perch_inference(infer_fn, dummy)
    print(f"  Perch shapes: embeddings={test_emb.shape}, logits={test_logits.shape}")

    # Find test soundscapes
    audio_files = sorted(
        list(test_dir.glob("*.ogg")) + list(test_dir.glob("*.wav")) + list(test_dir.glob("*.flac"))
    )
    if len(audio_files) == 0:
        train_dir = competition_dir / "train_soundscapes"
        if train_dir.exists():
            audio_files = sorted(train_dir.glob("*.ogg"))[:3]
            print(f"  No test soundscapes, using {len(audio_files)} train files")
    print(f"  Processing {len(audio_files)} test soundscapes")

    # ThreadPoolExecutor for I/O overlap (matching cache notebook pattern)
    def process_file(args):
        file_idx, audio_path = args
        if time.time() - START_TIME > TIMEOUT_SECONDS:
            return (file_idx, audio_path, None, None, True)

        waveform_batch = load_and_window_audio(audio_path)
        embeddings, logits = run_perch_inference(infer_fn, waveform_batch)
        mapped_scores = compute_perch_mapped_scores(logits, perch_species_mapping)
        return file_idx, audio_path, embeddings, mapped_scores, False

    all_embeddings = []
    all_mapped_scores = []
    test_meta_rows = []
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
                test_meta_rows.append(
                    {
                        "soundscape_id": soundscape_id,
                        "end_time": (w + 1) * WINDOW_SECONDS,
                        "site": site,
                        "hour": hour,
                    }
                )

            n_done = len(all_embeddings)
            elapsed = time.time() - START_TIME
            if n_done % 50 == 0 or n_done == 1 or n_done == len(audio_files):
                remaining = TIMEOUT_SECONDS - elapsed
                print(
                    f"  [{n_done}/{len(audio_files)}] {elapsed / 60:.1f}min elapsed, "
                    f"{remaining / 60:.1f}min remaining"
                )

    if n_timed_out > 0:
        print(f"  *** TIMEOUT: {n_timed_out} files skipped ***")

    if len(all_embeddings) == 0:
        print("  WARNING: No files processed!")
        test_emb_all = np.zeros((0, 1536), dtype=np.float32)
        test_scores_all = np.zeros((0, N_SPECIES_2026), dtype=np.float32)
    else:
        test_emb_all = np.concatenate(all_embeddings, axis=0)
        test_scores_all = np.concatenate(all_mapped_scores, axis=0)

    print(f"  Test: embeddings {test_emb_all.shape}, scores {test_scores_all.shape}")
    print(f"  Phase 2 time: {(time.time() - t_phase2) / 60:.1f}min")

    # -------------------------------------------------------------------
    # Phase 3: Apply probes + priors + smoothing
    # -------------------------------------------------------------------
    print("\n--- Phase 3: Apply Probes + Post-Processing ---")
    t_phase3 = time.time()

    # Group by soundscape for per-file processing
    soundscape_order = list(dict.fromkeys(r["soundscape_id"] for r in test_meta_rows))
    print(f"  Soundscapes: {len(soundscape_order)}")

    # Build index lookup: soundscape_id → list of row indices
    soundscape_indices = {}
    for i, row in enumerate(test_meta_rows):
        sid = row["soundscape_id"]
        if sid not in soundscape_indices:
            soundscape_indices[sid] = []
        soundscape_indices[sid].append(i)

    all_predictions = []

    for _sc_idx, soundscape_id in enumerate(soundscape_order):
        indices = soundscape_indices[soundscape_id]
        file_embeddings = test_emb_all[indices]
        file_mapped_scores = test_scores_all[indices]

        site = test_meta_rows[indices[0]]["site"]
        hour = test_meta_rows[indices[0]]["hour"]

        # PCA transform
        pca_embeddings = pca_transformer.transform(file_embeddings)

        # Apply probes
        perch_probs, _ = apply_perch_probes(
            pca_embeddings, file_mapped_scores, probe_models, probe_alphas, birdclef_species
        )

        # Apply metadata priors
        if metadata_priors is not None and (site is not None or hour is not None):
            perch_probs = apply_perch_metadata_priors(
                perch_probs,
                metadata_priors,
                site,
                hour,
                birdclef_species,
                taxa_classes,
                unmapped_set,
            )

        # Temporal smoothing
        perch_probs = apply_perch_temporal_smoothing(
            perch_probs,
            taxa_classes,
            birdclef_species,
            alpha_event=SMOOTHING_ALPHA_EVENT,
            alpha_texture=SMOOTHING_ALPHA_TEXTURE,
        )

        # Temperature
        if PERCH_TEMPERATURE != 1.0:
            perch_probs = _apply_temperature(perch_probs, PERCH_TEMPERATURE)

        # Build prediction rows
        for seg_idx, row_meta in enumerate(test_meta_rows[indices[0] : indices[-1] + 1]):
            if seg_idx >= perch_probs.shape[0]:
                break
            row_id = f"{soundscape_id}_{row_meta['end_time']}"
            row = {"row_id": row_id}
            for sp_idx, species in enumerate(birdclef_species):
                row[species] = float(perch_probs[seg_idx, sp_idx])
            all_predictions.append(row)

    print(f"  Total predictions: {len(all_predictions)} rows")
    print(f"  Phase 3 time: {time.time() - t_phase3:.1f}s")

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
    print(
        f"  Phase 1 (probes): {time.time() - t_phase1 - (t_phase2 - t_phase1) - (t_phase3 - t_phase2):.1f}s"
    )
    print(f"  Phase 2 (Perch): {(t_phase3 - t_phase2) / 60:.1f}min")
    print(f"  Phase 3 (scoring): {time.time() - t_phase3:.1f}s")


if __name__ == "__main__":
    main()
