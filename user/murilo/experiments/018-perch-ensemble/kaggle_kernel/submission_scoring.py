#!/usr/bin/env python3
"""BirdCLEF 2026 Scoring Kernel (Notebook B).

Loads pre-computed Perch embeddings + mapped scores from the cache kernel
(Notebook A), applies post-processing (PCA → probes → priors → smoothing),
and optionally runs SED inference for ensemble blending.

This is the second stage of a two-notebook chain:
  Notebook A (cache): Perch v2 inference → embeddings + mapped_scores
  Notebook B (this):  Load cache → score → submit

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

# Install onnxruntime only when SED is needed
if not PERCH_ONLY:
    DEPS_DIR = _find_input_early("birdclef-deps")
    print(f"DEPS_DIR resolved to: {DEPS_DIR} (exists={os.path.exists(DEPS_DIR)})")

    if os.path.exists(DEPS_DIR):
        print(f"Installing deps from {DEPS_DIR}")
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
    else:
        print(f"WARNING: deps dir not found at {DEPS_DIR}, onnxruntime may not be available")

# ---------------------------------------------------------------------------
# Imports (after offline install)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

if not PERCH_ONLY:
    import onnxruntime as ort  # noqa: E402

# ---------------------------------------------------------------------------
# Constants and post-processing flags
# ---------------------------------------------------------------------------
ENABLE_TTA = True
ENABLE_SED_TEMPORAL_SMOOTHING = True
ENABLE_SED_TOPN = True

N_SPECIES_2026 = 234

# Kaggle dataset slugs
PERCH_CACHE_SLUG = "birdclef-2026-perch-cache-v1"
PERCH_ARTIFACTS_SLUG = "birdclef-2026-perch-ensemble-v1"
SED_MODEL_SLUG = "birdclef-2026-effnet-b0-sed-v4b-weights"


# ===========================================================================
# Path resolution
# ===========================================================================


def _find_input(slug: str) -> str | None:
    """Resolve Kaggle input path, supporting both old and new layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
        # kernel_sources output paths (Kaggle mounts under "notebooks", not "kernels")
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
    """Find Perch cache output from Notebook A.

    When Notebook B lists Notebook A in kernel_sources, Kaggle mounts A's
    /kaggle/working/ output at /kaggle/input/notebooks/<user>/<slug>/.
    """
    path = _find_input(PERCH_CACHE_SLUG)
    if path:
        p = Path(path)
        # Verify expected files exist
        if (p / "perch_cache.npz").exists():
            return p
        # Check subdirectories (Kaggle may nest output)
        for sub in p.iterdir():
            if sub.is_dir() and (sub / "perch_cache.npz").exists():
                return sub

    # Brute-force: walk /kaggle/input/notebooks/ looking for perch_cache.npz
    notebooks_dir = Path("/kaggle/input/notebooks")
    if notebooks_dir.exists():
        print(f"  DEBUG: Walking {notebooks_dir} to find perch_cache.npz...")
        for root, _dirs, files in os.walk(str(notebooks_dir)):
            if "perch_cache.npz" in files:
                print(f"  DEBUG: Found at {root}")
                return Path(root)
        # Print what we see for debugging
        print(f"  DEBUG: perch_cache.npz not found. Contents of {notebooks_dir}:")
        for d1 in sorted(notebooks_dir.iterdir()):
            print(f"    {d1.name}/")
            if d1.is_dir():
                for d2 in sorted(d1.iterdir()):
                    print(f"      {d2.name}/")
                    if d2.is_dir():
                        for d3 in sorted(d2.iterdir()):
                            print(f"        {d3.name}")

    # Local testing fallback
    local = Path(os.environ.get("PERCH_CACHE_DIR", "predictions"))
    if (local / "perch_cache.npz").exists():
        return local

    return None


def find_perch_artifacts_dir() -> Path | None:
    """Find Perch ensemble artifacts directory (probes, PCA, priors)."""
    path = _find_input(PERCH_ARTIFACTS_SLUG)
    if path and os.path.exists(path):
        return Path(path)

    # Local testing fallback
    local = Path("artifacts/perch_probes")
    if local.exists():
        return local
    local2 = Path("artifacts")
    if (local2 / "perch_species_mapping.json").exists():
        return local2

    # Shared storage fallback
    shared = Path(
        "/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
    )
    if shared.exists():
        return shared

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
    """Load cached Perch embeddings + mapped scores + metadata from Notebook A.

    Returns:
        embeddings: (N, 1536) float32
        mapped_scores: (N, 234) float32
        meta_df: DataFrame with soundscape_id, end_time, site, hour, file_idx, window_idx
    """
    npz = np.load(cache_dir / "perch_cache.npz")
    embeddings = npz["embeddings"]
    mapped_scores = npz["mapped_scores"]
    meta_df = pd.read_parquet(cache_dir / "perch_meta.parquet")

    print(f"  Cache loaded: embeddings {embeddings.shape}, mapped_scores {mapped_scores.shape}")
    print(f"  Metadata: {len(meta_df)} rows, {meta_df['soundscape_id'].nunique()} soundscapes")

    return embeddings, mapped_scores, meta_df


# ===========================================================================
# Section 1: SED Inference (EfficientNet-B0 ONNX)
# ===========================================================================


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# ---------------------------------------------------------------------------
# Mel spectrogram computation
# ---------------------------------------------------------------------------


def compute_full_signal_mel(audio, mel_cfg):
    """Compute mel spectrogram for an entire audio signal (e.g. 60s).

    Returns the raw normalized mel as (n_mels, total_frames) WITHOUT
    batch/channel dims or frame truncation. Callers slice frames as needed.
    """
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
    """Slice a window from a full-signal mel, padding if needed.

    Returns (1, 1, n_mels, target_frames) ready for ONNX.
    """
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


# ---------------------------------------------------------------------------
# ONNX inference helpers
# ---------------------------------------------------------------------------


def _run_onnx_batch(session, mel_batch, mel_cfg):
    """Run batched ONNX inference. mel_batch: (B, 1, n_mels, T)."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel_batch})[0]
    if mel_cfg.get("output_type") == "probs":
        return output
    else:
        return sigmoid(output)


# ---------------------------------------------------------------------------
# SED post-processing
# ---------------------------------------------------------------------------


def _apply_temperature(probs_matrix, temperature):
    """Temperature scaling in logit space."""
    if temperature is None or temperature == 1.0:
        return probs_matrix

    eps = 1e-7
    p = np.clip(probs_matrix, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    scaled_logits = logits / temperature
    return sigmoid(scaled_logits)


def _apply_temporal_smoothing(probs_matrix, taxa_mapping=None):
    """Taxa-aware temporal smoothing across consecutive segments.

    When taxa_mapping is provided, applies different strategies by taxa:
    - Aves/Mammalia/Reptilia (events): local-max propagation (alpha=0.15)
    - Amphibia/Insecta (textures): average-neighbor smoothing (alpha=0.35)

    Fallback (no taxa_mapping): Gaussian kernel [0.1, 0.2, 0.4, 0.2, 0.1].
    """
    n_segments, n_species = probs_matrix.shape
    if n_segments < 3:
        return probs_matrix

    smoothed = np.copy(probs_matrix)

    if taxa_mapping is None:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for c in range(n_species):
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
        return smoothed

    # Build index sets by ecological behavior
    event_indices = set()
    texture_indices = set()
    for taxa in ["aves", "mammalia", "reptilia"]:
        event_indices.update(taxa_mapping.get(taxa, []))
    for taxa in ["amphibia", "insecta"]:
        texture_indices.update(taxa_mapping.get(taxa, []))

    alpha_event = 0.15
    alpha_texture = 0.35

    for c in range(n_species):
        if c in event_indices:
            alpha = alpha_event
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha) * probs_matrix[t, c] + alpha * max(prev_val, next_val)
        elif c in texture_indices:
            alpha = alpha_texture
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
    """TopN post-processing: multiply each segment by file-level max per species."""
    file_max = probs_matrix.max(axis=0, keepdims=True)
    return probs_matrix * file_max


# ---------------------------------------------------------------------------
# SED soundscape metadata parsing
# ---------------------------------------------------------------------------


def _parse_soundscape_metadata(filename):
    """Parse site and hour from soundscape filename.

    Format: BC2026_{Train|Test}_NNNN_SXX_YYYYMMDD_HHMMSS.ogg
    Returns: (site, hour) or (None, None) if parsing fails.
    """
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


def _apply_sed_metadata_priors(probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map):
    """Apply site/hour metadata priors multiplicatively for SED.

    probs *= prior ** lambda, where lambda varies by taxa.
    Fallback hierarchy: site+hour -> site -> hour -> global.
    """
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


# ---------------------------------------------------------------------------
# SED full-signal inference
# ---------------------------------------------------------------------------


def run_sed_inference(
    session, audio_path, mel_cfg, species_mapping, taxa_mapping=None, sed_metadata_priors=None
):
    """Process one soundscape using full-signal mel (compute STFT once, slice frames).

    Returns:
        sed_probs: (n_segments, n_model_outputs) probability matrix
        segment_times: list of end times for each segment
    """
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

    # Batched ONNX inference
    mel_batch = np.concatenate([s[2] for s in all_slices], axis=0)
    all_probs = _run_onnx_batch(session, mel_batch, mel_cfg)

    # Weighted average for TTA
    probs_matrix = np.zeros((n_segments, all_probs.shape[1]), dtype=np.float32)
    for i, (seg_idx, weight, _) in enumerate(all_slices):
        probs_matrix[seg_idx] += weight * all_probs[i]

    # SED post-processing pipeline
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
    """Map SED model output indices to 234 BirdCLEF 2026 species.

    Returns: (n_segments, 234) array with species probabilities.
    """
    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    n_segments = probs_matrix.shape[0]
    n_species = len(species_2026)

    mapped = np.zeros((n_segments, n_species), dtype=np.float32)
    species_to_col = {sp: i for i, sp in enumerate(species_2026)}

    for idx_str, name_2026 in idx_map.items():
        col_idx = species_to_col.get(name_2026)
        if col_idx is not None:
            mapped[:, col_idx] = probs_matrix[:, int(idx_str)]

    return mapped


# ===========================================================================
# Section 2: Perch Post-Processing (applied to cached embeddings/scores)
# ===========================================================================


def build_perch_features_for_file(pca_embeddings, mapped_scores, species_idx):
    """Build feature vector for a single species across all windows in one file.

    Features per (window, species):
    - PCA embedding (32 dims)
    - Mapped Perch score for this species (1)
    - Previous window score (1)
    - Next window score (1)
    - File mean score (1)
    - File max score (1)
    Total: 37 features
    """
    n_windows = pca_embeddings.shape[0]
    species_scores = mapped_scores[:, species_idx]

    prev_scores = np.zeros(n_windows, dtype=np.float32)
    next_scores = np.zeros(n_windows, dtype=np.float32)

    if n_windows > 1:
        prev_scores[1:] = species_scores[:-1]
        next_scores[:-1] = species_scores[1:]

    file_mean = species_scores.mean()
    file_max = species_scores.max()

    features = np.column_stack(
        [
            pca_embeddings,
            species_scores,
            prev_scores,
            next_scores,
            np.full(n_windows, file_mean, dtype=np.float32),
            np.full(n_windows, file_max, dtype=np.float32),
        ]
    )

    return features


def apply_perch_probes(pca_embeddings, mapped_scores, probe_models, probe_alphas, birdclef_species):
    """Apply per-species LogReg probes to refine Perch base scores.

    For species with a trained probe:
      probe_score = probe.predict_proba(features)[:, 1]
      alpha = probe_alphas[species]
      final_score = (1 - alpha) * base_score + alpha * probe_score

    For species without a probe: use base_score as-is.
    """
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
    perch_probs, metadata_priors, site, hour, birdclef_species, taxa_classes, unmapped_set, config
):
    """Apply Bayesian metadata priors to Perch predictions.

    Uses site/hour/interaction prior tables with taxa-aware lambda weights:
    - Aves/Mammalia/Reptilia (events): lambda_event (0.4)
    - Amphibia/Insecta (textures): lambda_texture (1.0)
    - Genus-proxy species: lambda_proxy_texture (0.8)
    """
    if metadata_priors is None:
        return perch_probs

    n_segments, n_species = perch_probs.shape
    lambda_event = config.get("lambda_event", 0.4)
    lambda_texture = config.get("lambda_texture", 1.0)
    lambda_proxy_texture = config.get("lambda_proxy_texture", 0.8)

    genus_proxy_set = set(config.get("genus_proxy_species", []))

    global_prior = np.array(metadata_priors.get("global_prior", []), dtype=np.float32)
    site_priors = metadata_priors.get("site_priors", {})
    hour_priors = metadata_priors.get("hour_priors", {})
    interaction_priors = metadata_priors.get("interaction_priors", {})

    if len(global_prior) == 0:
        return perch_probs

    site_str = site if site else None
    hour_str = str(hour) if hour is not None else None
    interaction_key = f"{site_str}_{hour_str}" if site_str and hour_str else None

    # Fallback hierarchy: interaction -> site -> hour -> global
    prior_vector = global_prior.copy()
    if interaction_key and interaction_key in interaction_priors:
        prior_vector = np.array(interaction_priors[interaction_key], dtype=np.float32)
    elif site_str and site_str in site_priors:
        prior_vector = np.array(site_priors[site_str], dtype=np.float32)
    elif hour_str and hour_str in hour_priors:
        prior_vector = np.array(hour_priors[hour_str], dtype=np.float32)

    if len(prior_vector) != n_species:
        return perch_probs

    # Build lambda vector per species
    lambda_vector = np.full(n_species, lambda_event, dtype=np.float32)
    for sp_idx, species in enumerate(birdclef_species):
        if species in unmapped_set:
            lambda_vector[sp_idx] = 0.0
            continue

        taxa = taxa_classes.get(species, "Aves")
        if species in genus_proxy_set:
            lambda_vector[sp_idx] = lambda_proxy_texture
        elif taxa in ("Amphibia", "Insecta"):
            lambda_vector[sp_idx] = lambda_texture
        elif taxa in ("Mammalia", "Reptilia"):
            lambda_vector[sp_idx] = lambda_event * 0.5

    prior_multiplier = np.power(np.maximum(prior_vector, 1e-10), lambda_vector)
    return perch_probs * prior_multiplier[np.newaxis, :]


def apply_perch_temporal_smoothing(
    perch_probs, taxa_classes, birdclef_species, alpha_event=0.15, alpha_texture=0.35
):
    """Taxa-aware temporal smoothing for Perch predictions.

    - Events (Aves/Mammalia/Reptilia): local-max propagation
    - Textures (Amphibia/Insecta): average-neighbor smoothing
    """
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
    """Blend SED and Perch predictions.

    For mapped species: final = sed_weight * sed + perch_weight * perch
    For unmapped species (25 insect sonotypes): final = sed (Perch has no signal)
    """
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
    # Resolve all input paths
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

    # Perch cache (from Notebook A)
    perch_cache_dir = find_perch_cache_dir()
    perch_cache_available = perch_cache_dir is not None

    # Perch artifacts (probes, PCA, priors)
    perch_artifacts_dir = find_perch_artifacts_dir()
    perch_available = perch_cache_available and perch_artifacts_dir is not None

    # SED model (skip in PERCH_ONLY mode)
    if PERCH_ONLY:
        sed_weights_dir = None
        sed_available = False
    else:
        sed_weights_dir = find_sed_weights_dir()
        sed_available = sed_weights_dir is not None and (sed_weights_dir / "model.onnx").exists()

    print(f"\nCompetition data: {competition_dir}")
    print(f"Perch cache: {perch_cache_dir} (available={perch_cache_available})")
    print(f"Perch artifacts: {perch_artifacts_dir} (found={perch_artifacts_dir is not None})")
    print(f"SED weights: {sed_weights_dir} (available={sed_available})")
    print(f"PERCH_ONLY: {PERCH_ONLY}")

    if not perch_available and not sed_available:
        raise FileNotFoundError(
            "Neither Perch cache nor SED model found! Need at least one for inference."
        )

    # -------------------------------------------------------------------
    # Load Perch cache + artifacts
    # -------------------------------------------------------------------
    embeddings_all = None
    mapped_scores_all = None
    meta_df = None
    perch_species_mapping = None
    pca_transformer = None
    probe_models = None
    probe_alphas = None
    perch_metadata_priors = None
    perch_config = None
    perch_taxa_classes = None
    perch_birdclef_species = None
    unmapped_set = set()
    unmapped_indices = set()

    if perch_available:
        print("\n--- Loading Perch Cache + Artifacts ---")
        t_load = time.time()

        # Load cache
        embeddings_all, mapped_scores_all, meta_df = load_perch_cache(perch_cache_dir)

        # Load species mapping
        perch_mapping_path = None
        for name in ["perch_species_mapping.json", "species_mapping.json"]:
            p = perch_artifacts_dir / name
            if p.exists():
                perch_mapping_path = p
                break

        if perch_mapping_path is not None:
            with open(perch_mapping_path) as f:
                perch_species_mapping = json.load(f)
            perch_birdclef_species = perch_species_mapping["birdclef_species"]
            perch_taxa_classes = perch_species_mapping.get("taxa_classes", {})
            unmapped_set = set(perch_species_mapping.get("unmapped", []))
            n_direct = len(perch_species_mapping.get("direct_map", {}))
            n_genus = len(perch_species_mapping.get("genus_proxy_map", {}))
            print(
                f"  Species mapping: {n_direct} direct, {n_genus} genus proxy, "
                f"{len(unmapped_set)} unmapped"
            )
            species_to_idx = {sp: i for i, sp in enumerate(perch_birdclef_species)}
            unmapped_indices = {species_to_idx[sp] for sp in unmapped_set if sp in species_to_idx}
        else:
            print(f"  WARNING: Species mapping not found in {perch_artifacts_dir}")
            perch_available = False

    if perch_available:
        # Load PCA transformer
        pca_path = perch_artifacts_dir / "pca_transformer.pkl"
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca_transformer = pickle.load(f)
            print(f"  PCA transformer: {pca_transformer.n_components} components")
        else:
            print(f"  WARNING: PCA transformer not found at {pca_path}")
            perch_available = False

    if perch_available:
        # Load probe models
        probes_path = perch_artifacts_dir / "probe_models.pkl"
        if probes_path.exists():
            with open(probes_path, "rb") as f:
                probe_models = pickle.load(f)
            print(f"  Probe models: {len(probe_models)} species")
        else:
            print(f"  WARNING: Probe models not found at {probes_path}")
            probe_models = {}

        # Load probe alphas
        alphas_path = perch_artifacts_dir / "probe_alphas.json"
        if alphas_path.exists():
            with open(alphas_path) as f:
                probe_alphas = json.load(f)
            print(f"  Probe alphas: {len(probe_alphas)} species")
        else:
            probe_alphas = {}

        # Load metadata priors
        priors_path = perch_artifacts_dir / "metadata_priors.json"
        if priors_path.exists():
            with open(priors_path) as f:
                perch_metadata_priors = json.load(f)
            print(
                f"  Metadata priors: {len(perch_metadata_priors.get('unique_sites', []))} sites, "
                f"{len(perch_metadata_priors.get('unique_hours', []))} hours"
            )
        else:
            perch_metadata_priors = None

        # Load config
        config_path = perch_artifacts_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                perch_config = json.load(f)
            print(f"  Config loaded (feature_dim={perch_config.get('feature_dim', '?')})")
        else:
            perch_config = {}

        print(f"  Perch artifacts loaded in {time.time() - t_load:.1f}s")

    # -------------------------------------------------------------------
    # Load SED artifacts
    # -------------------------------------------------------------------
    onnx_session = None
    sed_mel_cfg = None
    sed_species_mapping = None
    sed_taxa_mapping = None
    sed_metadata_priors = None

    if sed_available:
        print("\n--- Loading SED Model Artifacts ---")
        onnx_path = sed_weights_dir / "model.onnx"
        mel_config_path = sed_weights_dir / "mel_config.json"
        species_mapping_path = sed_weights_dir / "species_mapping.json"

        with open(mel_config_path) as f:
            sed_mel_cfg = json.load(f)
        with open(species_mapping_path) as f:
            sed_species_mapping = json.load(f)

        print(f"  Mel type: {sed_mel_cfg.get('type', 'mel')}, SR: {sed_mel_cfg['sr']}")

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
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"  ONNX session created ({time.time() - START_TIME:.1f}s)")

    # -------------------------------------------------------------------
    # Determine ensemble mode
    # -------------------------------------------------------------------
    if sed_available and perch_available:
        mode = "ensemble"
        ensemble_cfg = perch_config.get("ensemble", {})
        sed_weight = ensemble_cfg.get("sed_weight", 0.3)
        perch_weight = ensemble_cfg.get("perch_weight", 0.7)
        perch_temperature = ensemble_cfg.get("temperature", 1.10)
        smoothing_cfg = perch_config.get("smoothing", {})
        perch_alpha_event = smoothing_cfg.get("alpha_event", 0.15)
        perch_alpha_texture = smoothing_cfg.get("alpha_texture", 0.35)
        priors_cfg = perch_config.get("priors", {})
    elif perch_available:
        mode = "perch_only"
        sed_weight = 0.0
        perch_weight = 1.0
        ensemble_cfg = perch_config.get("ensemble", {})
        perch_temperature = ensemble_cfg.get("temperature", 1.10)
        smoothing_cfg = perch_config.get("smoothing", {})
        perch_alpha_event = smoothing_cfg.get("alpha_event", 0.15)
        perch_alpha_texture = smoothing_cfg.get("alpha_texture", 0.35)
        priors_cfg = perch_config.get("priors", {})
    elif sed_available:
        mode = "sed_only"
        sed_weight = 1.0
        perch_weight = 0.0
    else:
        raise RuntimeError("No models available for inference")

    print(f"\n--- Mode: {mode.upper()} ---")
    if mode == "ensemble":
        print(f"  SED weight: {sed_weight}, Perch weight: {perch_weight}")
    if mode in ("perch_only", "ensemble"):
        print(f"  Perch temperature: {perch_temperature}")
        print(f"  Smoothing: event={perch_alpha_event}, texture={perch_alpha_texture}")

    # -------------------------------------------------------------------
    # Determine species list and soundscape ordering
    # -------------------------------------------------------------------
    if perch_available and perch_birdclef_species is not None:
        species_list_2026 = perch_birdclef_species
    elif sed_available:
        species_list_2026 = sed_species_mapping["birdclef2026_species"]
    else:
        species_list_2026 = []

    n_species = len(species_list_2026)
    print(f"Species list: {n_species} species")

    # Group cache by soundscape for per-file processing
    if perch_available and meta_df is not None:
        soundscape_order = list(dict.fromkeys(meta_df["soundscape_id"]))
        print(f"Soundscapes in cache: {len(soundscape_order)}")
    else:
        soundscape_order = []

    # If SED-only mode, we need to find audio files ourselves
    if mode == "sed_only":
        audio_files = sorted(
            list(test_dir.glob("*.ogg"))
            + list(test_dir.glob("*.wav"))
            + list(test_dir.glob("*.flac"))
        )
        if len(audio_files) == 0:
            train_dir = competition_dir / "train_soundscapes"
            if train_dir.exists():
                audio_files = sorted(train_dir.glob("*.ogg"))[:3]
        soundscape_order = [f.stem for f in audio_files]
        soundscape_to_path = {f.stem: f for f in audio_files}
        print(f"Audio files found: {len(audio_files)}")
    else:
        # Build path lookup for SED in ensemble mode
        soundscape_to_path = {}
        if sed_available:
            audio_files = sorted(
                list(test_dir.glob("*.ogg"))
                + list(test_dir.glob("*.wav"))
                + list(test_dir.glob("*.flac"))
            )
            if len(audio_files) == 0:
                train_dir = competition_dir / "train_soundscapes"
                if train_dir.exists():
                    audio_files = sorted(train_dir.glob("*.ogg"))
            soundscape_to_path = {f.stem: f for f in audio_files}

    # -------------------------------------------------------------------
    # Process each soundscape
    # -------------------------------------------------------------------
    all_predictions = []
    timing_stats = {"perch_pp": [], "sed": [], "blend": [], "total": []}

    for sc_idx, soundscape_id in enumerate(soundscape_order):
        t_file_start = time.time()

        # ----- a. Perch post-processing (from cache) -----
        perch_probs = None
        segment_times = None

        if perch_available and meta_df is not None:
            t_pp = time.time()

            # Get this file's data from cache
            file_mask = meta_df["soundscape_id"] == soundscape_id
            file_rows = meta_df[file_mask].sort_values("window_idx")
            file_indices = file_rows.index.values

            file_embeddings = embeddings_all[file_indices]  # (n_windows, 1536)
            file_mapped_scores = mapped_scores_all[file_indices]  # (n_windows, 234)

            site = file_rows.iloc[0]["site"]
            hour = file_rows.iloc[0]["hour"]
            segment_times = file_rows["end_time"].tolist()

            # PCA transform
            pca_embeddings = pca_transformer.transform(file_embeddings)

            # Apply LogReg probes
            perch_probs, n_probed = apply_perch_probes(
                pca_embeddings,
                file_mapped_scores,
                probe_models,
                probe_alphas,
                perch_birdclef_species,
            )

            # Apply metadata priors
            if perch_metadata_priors is not None and (site is not None or hour is not None):
                perch_probs = apply_perch_metadata_priors(
                    perch_probs,
                    perch_metadata_priors,
                    site,
                    hour,
                    perch_birdclef_species,
                    perch_taxa_classes,
                    unmapped_set,
                    priors_cfg,
                )

            # Taxa-aware temporal smoothing
            perch_probs = apply_perch_temporal_smoothing(
                perch_probs,
                perch_taxa_classes,
                perch_birdclef_species,
                alpha_event=perch_alpha_event,
                alpha_texture=perch_alpha_texture,
            )

            # Temperature scaling
            if perch_temperature != 1.0:
                perch_probs = _apply_temperature(perch_probs, perch_temperature)

            timing_stats["perch_pp"].append(time.time() - t_pp)

        # ----- b. SED inference (from audio) -----
        sed_probs_mapped = None

        if sed_available and soundscape_id in soundscape_to_path:
            t_sed = time.time()
            audio_path = soundscape_to_path[soundscape_id]
            sed_probs_raw, sed_segment_times = run_sed_inference(
                onnx_session,
                str(audio_path),
                sed_mel_cfg,
                sed_species_mapping,
                taxa_mapping=sed_taxa_mapping,
                sed_metadata_priors=sed_metadata_priors,
            )
            sed_probs_mapped = map_sed_to_species(sed_probs_raw, sed_species_mapping)

            # Use SED segment times if Perch not available
            if segment_times is None:
                segment_times = sed_segment_times

            timing_stats["sed"].append(time.time() - t_sed)

        # ----- c. Ensemble blend -----
        t_blend = time.time()

        if segment_times is None:
            # No data for this soundscape (shouldn't happen)
            print(f"  WARNING: No segment times for {soundscape_id}, skipping")
            continue

        n_segments = len(segment_times)

        if mode == "ensemble":
            # Ensure shapes match
            if sed_probs_mapped is not None and perch_probs is not None:
                min_seg = min(sed_probs_mapped.shape[0], perch_probs.shape[0])
                sed_probs_mapped = sed_probs_mapped[:min_seg]
                perch_probs = perch_probs[:min_seg]
                segment_times = segment_times[:min_seg]
                n_segments = min_seg

                final_probs = blend_ensemble(
                    sed_probs_mapped,
                    perch_probs,
                    unmapped_indices,
                    sed_weight=sed_weight,
                    perch_weight=perch_weight,
                )
            elif perch_probs is not None:
                final_probs = perch_probs
            elif sed_probs_mapped is not None:
                final_probs = sed_probs_mapped
            else:
                continue
        elif mode == "perch_only":
            final_probs = perch_probs
        elif mode == "sed_only":
            final_probs = sed_probs_mapped
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        timing_stats["blend"].append(time.time() - t_blend)

        # ----- d. Build prediction rows -----
        for seg_idx, end_time in enumerate(segment_times):
            if seg_idx >= final_probs.shape[0]:
                break
            row_id = f"{soundscape_id}_{end_time}"
            row = {"row_id": row_id}
            for sp_idx, species in enumerate(species_list_2026):
                row[species] = float(final_probs[seg_idx, sp_idx])
            all_predictions.append(row)

        t_file_elapsed = time.time() - t_file_start
        timing_stats["total"].append(t_file_elapsed)

        if (sc_idx + 1) % 50 == 0 or sc_idx == 0 or (sc_idx + 1) == len(soundscape_order):
            print(
                f"  [{sc_idx + 1}/{len(soundscape_order)}] {soundscape_id}: "
                f"{n_segments} segments ({t_file_elapsed:.2f}s)"
            )

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
    print(f"Mode: {mode} | SED weight: {sed_weight} | Perch weight: {perch_weight}")


if __name__ == "__main__":
    main()
