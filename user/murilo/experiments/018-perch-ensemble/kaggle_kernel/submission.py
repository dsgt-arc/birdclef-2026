#!/usr/bin/env python3
"""BirdCLEF 2026 Perch Ensemble Kaggle Submission Kernel.

Self-contained Kaggle CPU kernel that ensembles:
  1. EfficientNet-B0 SED (ONNX) — our trained model on BirdCLEF 2026 focal + soundscapes
  2. Google Perch v2 (TF SavedModel) — frozen foundation model with LogReg probes

Pipeline per 60s soundscape:
  a. SED inference: full-signal mel + batched ONNX -> sed_probs (12, 234)
  b. Perch inference: 12x5s windows -> TF model -> embeddings + logits
  c. Perch post-processing: PCA + LogReg probes + metadata priors + smoothing
  d. Ensemble blend: final = sed_weight * sed + perch_weight * perch

For the 25 unmapped insect sonotypes (no Perch signal), uses SED-only predictions.

Graceful fallback: if Perch model or artifacts are unavailable, runs SED-only mode.
"""

# ---------------------------------------------------------------------------
# Offline dependency installation (runs on Kaggle before imports)
# Wheels are in a separate Kaggle dataset: birdclef-deps
# ---------------------------------------------------------------------------
import os
import subprocess
import sys

KAGGLE_USER = "murilogustineli"

# Mode: set to True for Phase A (Perch-only baseline), False for Phase B (ensemble)
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
    return candidates[0]  # fallback to old-style


# Debug: show what we see in /kaggle/input
if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")
print(f"PERCH_ONLY mode: {PERCH_ONLY}")

# Install onnxruntime only when SED is needed (not in PERCH_ONLY mode)
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
# Imports (after offline install, must use noqa: E402)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
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

# Number of BirdCLEF 2026 species
N_SPECIES_2026 = 234

# ---------------------------------------------------------------------------
# Kaggle path resolution
# ---------------------------------------------------------------------------

# SED model weights slug
SED_MODEL_SLUG = "birdclef-2026-effnet-b0-sed-v4b-weights"

# Perch ensemble artifacts slug
PERCH_ARTIFACTS_SLUG = "birdclef-2026-perch-ensemble-v1"

# Perch v2 model (from google/bird-vocalization-classifier)
# On Kaggle, model sources are mounted under /kaggle/input/
PERCH_MODEL_CANDIDATES = [
    # Kaggle model source mount points (various layouts)
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch-v2/1",
    "/kaggle/input/bird-vocalization-classifier/TensorFlow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier/TensorFlow2/perch-v2/1",
    "/kaggle/input/bird-vocalization-classifier",
    # Alternate Kaggle layouts
    "/kaggle/input/models/google/bird-vocalization-classifier/TensorFlow2/perch_v2/1",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/1",
]


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


def find_sed_weights_dir() -> Path | None:
    """Find SED model weights directory."""
    path = _find_input(SED_MODEL_SLUG)
    if path and os.path.exists(path):
        return Path(path)

    # Local testing fallback
    local = Path("artifacts/weights")
    if (local / "model.onnx").exists():
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

    return None


def find_perch_model_dir() -> Path | None:
    """Find Perch v2 TF SavedModel directory."""
    for candidate in PERCH_MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            # Check if it has saved_model.pb (could be in subdir)
            if (p / "saved_model.pb").exists():
                return p
            # Check one level deeper
            for subdir in sorted(p.iterdir()):
                if subdir.is_dir() and (subdir / "saved_model.pb").exists():
                    return subdir
            # If directory exists but no saved_model.pb found, still return it
            # TF will handle the error
            return p

    # Local testing fallback
    local = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2")
    if local.exists():
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
    # Local fallback
    return Path(os.environ.get("COMPETITION_DIR", "data/birdclef-2026"))


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
    This avoids redundant STFT computation for overlapping segments.
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
        # AmplitudeToDB -> standardize -> min-max [0,1] on the full signal.
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
    return mel.astype(np.float32)  # (n_mels, total_frames)


def _slice_mel_window(full_mel, start_frame, target_frames):
    """Slice a window from a full-signal mel, padding if needed.

    Returns (1, 1, n_mels, target_frames) ready for ONNX.
    """
    total_frames = full_mel.shape[1]

    # Handle negative start (before audio begins)
    if start_frame < 0:
        pad_left = -start_frame
        start_frame = 0
    else:
        pad_left = 0

    end_frame = start_frame + target_frames - pad_left
    if end_frame > total_frames:
        end_frame = total_frames

    window = full_mel[:, start_frame:end_frame]

    # Pad if needed (left for negative start, right for end of file)
    pad_right = target_frames - pad_left - window.shape[1]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, max(0, pad_right))), mode="constant")

    return window[np.newaxis, np.newaxis, :, :]  # (1, 1, n_mels, target_frames)


# ---------------------------------------------------------------------------
# ONNX inference helpers
# ---------------------------------------------------------------------------


def _run_onnx_batch(session, mel_batch, mel_cfg):
    """Run batched ONNX inference. mel_batch: (B, 1, n_mels, T)."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel_batch})[0]  # (B, n_classes)
    if mel_cfg.get("output_type") == "probs":
        return output
    else:
        return sigmoid(output)


# ---------------------------------------------------------------------------
# SED post-processing
# ---------------------------------------------------------------------------


def _apply_temperature(probs_matrix, temperature):
    """Temperature scaling in logit space.

    Converts probabilities to logits, scales by 1/T, converts back.
    T > 1 softens (reduces overconfidence), T < 1 sharpens.
    """
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
      x_new = (1-a)*x + a*max(prev, next) -- preserves transient peaks
    - Amphibia/Insecta (textures): average-neighbor smoothing (alpha=0.35)
      x_new = (1-a)*x + a*mean(prev, next) -- solidifies continuous signals

    Fallback (no taxa_mapping): Gaussian kernel [0.1, 0.2, 0.4, 0.2, 0.1].
    """
    n_segments, n_species = probs_matrix.shape
    if n_segments < 3:
        return probs_matrix

    smoothed = np.copy(probs_matrix)

    if taxa_mapping is None:
        # Fallback: original uniform Gaussian kernel
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for c in range(n_species):
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
        return smoothed

    # Build index sets by ecological behavior
    event_indices = set()  # transient calls
    texture_indices = set()  # continuous sounds
    for taxa in ["aves", "mammalia", "reptilia"]:
        event_indices.update(taxa_mapping.get(taxa, []))
    for taxa in ["amphibia", "insecta"]:
        texture_indices.update(taxa_mapping.get(taxa, []))

    alpha_event = 0.15
    alpha_texture = 0.35

    for c in range(n_species):
        if c in event_indices:
            # Local-max propagation: preserves transient bird call peaks
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha_event) * probs_matrix[t, c] + alpha_event * max(
                    prev_val, next_val
                )
        elif c in texture_indices:
            # Average-neighbor: solidifies continuous frog/insect textures
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha_texture) * probs_matrix[t, c] + alpha_texture * (
                    (prev_val + next_val) / 2
                )
        else:
            # Unknown taxa: use Gaussian kernel
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")

    return smoothed


def _apply_topn(probs_matrix):
    """TopN postprocessing: multiply each segment by file-level max per species.

    Boosts species that are consistently detected across the file and suppresses
    spurious single-segment detections. Confirmed +0.011 LB.
    """
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

    site = parts[3]  # e.g. "S08"
    try:
        hour = int(parts[5][:2])  # first 2 digits of HHMMSS
    except (ValueError, IndexError):
        return site, None

    return site, hour


def _apply_sed_metadata_priors(probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map):
    """Apply site/hour metadata priors multiplicatively for SED.

    probs *= prior ** lambda, where lambda varies by taxa:
    - Aves: 0.3 (birds somewhat mobile, moderate prior)
    - Amphibia/Insecta: 0.5 (strongly site/time dependent)
    - Mammalia/Reptilia: 0.2 (most mobile)

    Fallback hierarchy: site+hour -> site -> hour -> global.
    """
    if metadata_priors is None:
        return probs_matrix

    n_segments, n_species = probs_matrix.shape
    lambda_map = {"aves": 0.3, "amphibia": 0.5, "insecta": 0.5, "mammalia": 0.2, "reptilia": 0.2}

    # Build index-to-taxa lookup
    idx_to_taxa = {}
    if taxa_mapping:
        for taxa_class, indices in taxa_mapping.items():
            for idx in indices:
                idx_to_taxa[idx] = taxa_class

    # Prior lookup tables
    site_hour_priors = metadata_priors.get("site_hour_species", {})
    site_priors = metadata_priors.get("site_species", {})
    hour_priors = metadata_priors.get("hour_species", {})
    global_priors = metadata_priors.get("global_species", {})

    site_str = site if site else None
    hour_str = str(hour).zfill(2) if hour is not None else None

    # Build per-species prior multiplier vector
    prior_vector = np.ones(n_species, dtype=np.float32)
    for idx_str, species_code in idx_map.items():
        idx = int(idx_str)
        if idx >= n_species:
            continue

        # Fallback hierarchy: site+hour -> site -> hour -> global
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
            continue  # No prior data; leave at 1.0

        # Scale lambda by taxa class
        taxa = idx_to_taxa.get(idx, "aves")
        lam = lambda_map.get(taxa, 0.3)
        prior_vector[idx] = prior_val**lam

    # Apply multiplicatively to all segments
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
        soundscape_id: stem of the audio file
    """
    sr = mel_cfg["sr"]
    hop_length = mel_cfg["hop_length"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2
    target_frames = mel_cfg.get("target_frames", 1000)

    # 1. Load entire audio file once
    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    total_duration = len(audio) / sr

    # 2. Pad audio for context (negative start / end overflow)
    context_samples = int(context_pad * sr)
    tta_shift_samples = int(2.5 * sr)  # TTA shift = 2.5s
    pad_samples = context_samples + tta_shift_samples
    audio_padded = np.pad(audio, (pad_samples, pad_samples), mode="constant")

    # 3. Compute full-signal mel ONCE on the padded audio
    full_mel = compute_full_signal_mel(audio_padded, mel_cfg)

    # Frame offset corresponding to t=0 in the original audio
    origin_frame = int(pad_samples / hop_length)

    # 4. Build all mel slices for all segments + TTA shifts
    segment_times = []
    segment_start_s = 0.0
    all_slices = []  # list of (segment_idx, tta_weight, mel_slice)
    segment_idx = 0

    while segment_start_s < total_duration:
        # Center of inference window on 5s segment
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

    # 5. Batched ONNX inference
    mel_batch = np.concatenate([s[2] for s in all_slices], axis=0)  # (N, 1, n_mels, T)
    all_probs = _run_onnx_batch(session, mel_batch, mel_cfg)  # (N, n_classes)

    # 6. Weighted average for TTA
    probs_matrix = np.zeros((n_segments, all_probs.shape[1]), dtype=np.float32)
    for i, (seg_idx, weight, _) in enumerate(all_slices):
        probs_matrix[seg_idx] += weight * all_probs[i]

    # 7. SED post-processing pipeline
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
# Section 2: Perch v2 Inference
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

    # Try common inference methods
    infer_fn = None

    # Method 1: Check for 'infer_tf' method (common for Perch)
    if hasattr(model, "infer_tf"):
        infer_fn = model.infer_tf
        print("  Using model.infer_tf() for inference")
    # Method 2: Check for 'serving_default' signature
    elif "serving_default" in sigs:
        infer_fn = model.signatures["serving_default"]
        print("  Using model.signatures['serving_default'] for inference")
    # Method 3: Try calling model directly
    elif callable(model):
        infer_fn = model
        print("  Using model() directly for inference")
    else:
        # Method 4: Use first available signature
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

    # Perch v2 serving_default expects 'inputs' as keyword argument
    try:
        output = infer_fn(inputs=input_tensor)
    except TypeError:
        output = infer_fn(input_tensor)

    # Handle different output formats
    if isinstance(output, dict):
        # Perch v2 outputs: 'embedding', 'label', 'spectrogram', 'spatial_embedding'
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
            # Fall back: identify by shape
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
            # Offset by -1: labels.csv index -> model output index
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


# ---------------------------------------------------------------------------
# Perch post-processing
# ---------------------------------------------------------------------------


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

    # Temporal context within this file
    prev_scores = np.zeros(n_windows, dtype=np.float32)
    next_scores = np.zeros(n_windows, dtype=np.float32)

    if n_windows > 1:
        prev_scores[1:] = species_scores[:-1]
        next_scores[:-1] = species_scores[1:]

    file_mean = species_scores.mean()
    file_max = species_scores.max()

    features = np.column_stack(
        [
            pca_embeddings,  # 32 dims
            species_scores,  # 1
            prev_scores,  # 1
            next_scores,  # 1
            np.full(n_windows, file_mean, dtype=np.float32),  # 1
            np.full(n_windows, file_max, dtype=np.float32),  # 1
        ]
    )

    return features  # (n_windows, 37)


def apply_perch_probes(pca_embeddings, mapped_scores, probe_models, probe_alphas, birdclef_species):
    """Apply per-species LogReg probes to refine Perch base scores.

    For species with a trained probe:
      probe_score = probe.predict_proba(features)[:, 1]
      alpha = probe_alphas[species]
      final_score = (1 - alpha) * base_score + alpha * probe_score

    For species without a probe: use base_score as-is.
    """
    perch_probs = np.copy(mapped_scores)  # (n_windows, n_species)

    n_probed = 0
    for sp_idx, species in enumerate(birdclef_species):
        if species not in probe_models:
            continue

        probe = probe_models[species]
        alpha = probe_alphas.get(species, 0.25)

        # Build features for this species
        features = build_perch_features_for_file(pca_embeddings, mapped_scores, sp_idx)

        # Predict
        try:
            probe_score = probe.predict_proba(features)[:, 1]
        except Exception:
            # If probe fails (e.g., shape mismatch), keep base score
            continue

        # Dynamic alpha blend
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

    # Determine which species are genus proxies
    genus_proxy_set = set(config.get("genus_proxy_species", []))

    # Prior tables: vectors indexed by species position
    global_prior = np.array(metadata_priors.get("global_prior", []), dtype=np.float32)
    site_priors = metadata_priors.get("site_priors", {})
    hour_priors = metadata_priors.get("hour_priors", {})
    interaction_priors = metadata_priors.get("interaction_priors", {})

    if len(global_prior) == 0:
        return perch_probs

    # Find the best prior vector for this soundscape
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

    # Ensure prior_vector matches species count
    if len(prior_vector) != n_species:
        return perch_probs

    # Build lambda vector per species
    lambda_vector = np.full(n_species, lambda_event, dtype=np.float32)
    for sp_idx, species in enumerate(birdclef_species):
        if species in unmapped_set:
            lambda_vector[sp_idx] = 0.0  # No Perch prior for unmapped species
            continue

        taxa = taxa_classes.get(species, "Aves")
        if species in genus_proxy_set:
            lambda_vector[sp_idx] = lambda_proxy_texture
        elif taxa in ("Amphibia", "Insecta"):
            lambda_vector[sp_idx] = lambda_texture
        elif taxa in ("Mammalia", "Reptilia"):
            lambda_vector[sp_idx] = lambda_event * 0.5  # More mobile
        # else: default lambda_event for Aves

    # Apply: probs *= prior ** lambda
    prior_multiplier = np.power(
        np.maximum(prior_vector, 1e-10),
        lambda_vector,
    )
    return perch_probs * prior_multiplier[np.newaxis, :]


def apply_perch_temporal_smoothing(
    perch_probs, taxa_classes, birdclef_species, alpha_event=0.15, alpha_texture=0.35
):
    """Taxa-aware temporal smoothing for Perch predictions.

    Same logic as SED smoothing but applied to Perch probability matrix.
    - Events (Aves/Mammalia/Reptilia): local-max propagation
    - Textures (Amphibia/Insecta): average-neighbor smoothing
    """
    n_segments, n_species = perch_probs.shape
    if n_segments < 3:
        return perch_probs

    smoothed = np.copy(perch_probs)

    # Build taxa classification for each species column
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
            # Default: Gaussian kernel
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

    Args:
        sed_probs_mapped: (n_segments, 234) SED predictions mapped to species
        perch_probs: (n_segments, 234) Perch predictions
        unmapped_indices: set of species column indices that Perch cannot predict
        sed_weight: weight for SED predictions (default 0.3)
        perch_weight: weight for Perch predictions (default 0.7)

    Returns:
        final_probs: (n_segments, 234) blended predictions
    """
    final_probs = sed_weight * sed_probs_mapped + perch_weight * perch_probs

    # For unmapped species, use SED-only (Perch is zero anyway, but be explicit)
    for idx in unmapped_indices:
        final_probs[:, idx] = sed_probs_mapped[:, idx]

    return final_probs


# ===========================================================================
# Section 4: Main
# ===========================================================================


def main():
    start_time = time.time()

    print("=" * 70)
    print("BirdCLEF 2026 Perch Ensemble Submission")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Resolve all input paths
    # -------------------------------------------------------------------
    competition_dir = find_competition_dir()
    test_dir = competition_dir / "test_soundscapes"
    sample_sub_path = competition_dir / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")

    # Fallback for local testing
    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not sample_sub_path.exists():
        sample_sub_path = Path(os.environ.get("SAMPLE_SUBMISSION", "sample_submission.csv"))
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    # SED model (skip entirely in PERCH_ONLY mode)
    if PERCH_ONLY:
        sed_weights_dir = None
        sed_available = False
    else:
        sed_weights_dir = find_sed_weights_dir()
        sed_available = sed_weights_dir is not None and (sed_weights_dir / "model.onnx").exists()

    # Perch model
    perch_model_dir = find_perch_model_dir()
    perch_artifacts_dir = find_perch_artifacts_dir()
    perch_available = perch_model_dir is not None and perch_artifacts_dir is not None

    print(f"\nCompetition data: {competition_dir}")
    print(f"Test soundscapes: {test_dir}")
    print(f"SED weights: {sed_weights_dir} (available={sed_available})")
    print(f"Perch model: {perch_model_dir} (found={perch_model_dir is not None})")
    print(f"Perch artifacts: {perch_artifacts_dir} (found={perch_artifacts_dir is not None})")
    print(f"Perch available: {perch_available}")

    if not sed_available and not perch_available:
        raise FileNotFoundError(
            "Neither SED nor Perch model found! Need at least one model for inference."
        )

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
        idx_map = (
            sed_species_mapping.get("idx_to_name_2026")
            or sed_species_mapping["idx_2025_to_name_2026"]
        )
        print(f"  Species mapping: {len(idx_map)} mapped to 2026 columns")

        # Load optional pp2 artifacts
        taxa_path = sed_weights_dir / "taxa_mapping.json"
        if taxa_path.exists():
            with open(taxa_path) as f:
                sed_taxa_mapping = json.load(f)
            n_taxa_species = sum(len(v) for v in sed_taxa_mapping.values())
            print(
                f"  Taxa mapping: {n_taxa_species} species across {len(sed_taxa_mapping)} classes"
            )

        priors_path = sed_weights_dir / "metadata_priors.json"
        if priors_path.exists():
            with open(priors_path) as f:
                sed_metadata_priors = json.load(f)
            print("  SED metadata priors loaded")

        # Create ONNX session
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = 1
        onnx_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        print(
            f"  ONNX session created (threads={sess_opts.intra_op_num_threads}, "
            f"{time.time() - start_time:.1f}s)"
        )

    # -------------------------------------------------------------------
    # Load Perch artifacts
    # -------------------------------------------------------------------
    perch_infer_fn = None
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
        print("\n--- Loading Perch Model & Artifacts ---")
        t_perch_load = time.time()

        # Load Perch species mapping
        perch_mapping_path = perch_artifacts_dir / "perch_species_mapping.json"
        if not perch_mapping_path.exists():
            # Try without prefix
            for candidate_name in ["perch_species_mapping.json", "species_mapping.json"]:
                p = perch_artifacts_dir / candidate_name
                if p.exists():
                    perch_mapping_path = p
                    break

        if perch_mapping_path.exists():
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

            # Build unmapped index set
            species_to_idx = {sp: i for i, sp in enumerate(perch_birdclef_species)}
            unmapped_indices = {species_to_idx[sp] for sp in unmapped_set if sp in species_to_idx}
        else:
            print(f"  WARNING: Perch species mapping not found at {perch_mapping_path}")
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

        # Load Perch TF model
        try:
            _, perch_infer_fn = load_perch_model(str(perch_model_dir))

            # Probe model with dummy input to confirm shapes
            dummy = np.zeros((1, 160000), dtype=np.float32)
            test_emb, test_logits = run_perch_inference(perch_infer_fn, dummy)
            print(f"  Perch output shapes: embeddings={test_emb.shape}, logits={test_logits.shape}")
            print(f"  Perch model loaded in {time.time() - t_perch_load:.1f}s")
        except Exception as e:
            print(f"  ERROR loading Perch model: {e}")
            print("  Falling back to SED-only mode")
            perch_available = False

    # -------------------------------------------------------------------
    # Determine ensemble mode
    # -------------------------------------------------------------------
    if sed_available and perch_available:
        mode = "ensemble"
        # Get ensemble weights from Perch config
        ensemble_cfg = perch_config.get("ensemble", {})
        sed_weight = ensemble_cfg.get("sed_weight", 0.3)
        perch_weight = ensemble_cfg.get("perch_weight", 0.7)
        perch_temperature = ensemble_cfg.get("temperature", 1.10)
        smoothing_cfg = perch_config.get("smoothing", {})
        perch_alpha_event = smoothing_cfg.get("alpha_event", 0.15)
        perch_alpha_texture = smoothing_cfg.get("alpha_texture", 0.35)
        priors_cfg = perch_config.get("priors", {})
    elif sed_available:
        mode = "sed_only"
        sed_weight = 1.0
        perch_weight = 0.0
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
    else:
        raise RuntimeError("No models available for inference")

    print(f"\n--- Ensemble Mode: {mode.upper()} ---")
    if mode == "ensemble":
        print(f"  SED weight: {sed_weight}, Perch weight: {perch_weight}")
        print(f"  Perch temperature: {perch_temperature}")
        print(f"  Perch smoothing: event={perch_alpha_event}, texture={perch_alpha_texture}")

    # -------------------------------------------------------------------
    # Find soundscape files
    # -------------------------------------------------------------------
    audio_files = sorted(
        list(test_dir.glob("*.ogg")) + list(test_dir.glob("*.wav")) + list(test_dir.glob("*.flac"))
    )

    # Fallback: use train_soundscapes for validation
    if len(audio_files) == 0:
        train_dir = competition_dir / "train_soundscapes"
        if train_dir.exists():
            audio_files = sorted(train_dir.glob("*.ogg"))[:2]
            print(f"No test soundscapes found, using {len(audio_files)} train files for validation")

    print(f"\nProcessing {len(audio_files)} soundscapes from {test_dir}")

    # Determine species list for output columns
    if perch_available and perch_birdclef_species is not None:
        species_list_2026 = perch_birdclef_species
    elif sed_available:
        species_list_2026 = sed_species_mapping["birdclef2026_species"]
    else:
        species_list_2026 = []

    n_species = len(species_list_2026)
    print(f"Species list: {n_species} species")

    # -------------------------------------------------------------------
    # Process each soundscape
    # -------------------------------------------------------------------
    all_predictions = []
    timing_stats = {"sed": [], "perch": [], "blend": [], "total": []}

    for file_idx, audio_path in enumerate(audio_files):
        t_file_start = time.time()
        soundscape_id = audio_path.stem
        site, hour = _parse_soundscape_metadata(str(audio_path))

        # ----- a. SED inference -----
        sed_probs_mapped = None
        segment_times = None

        if sed_available:
            t_sed = time.time()
            sed_probs_raw, segment_times = run_sed_inference(
                onnx_session,
                str(audio_path),
                sed_mel_cfg,
                sed_species_mapping,
                taxa_mapping=sed_taxa_mapping,
                sed_metadata_priors=sed_metadata_priors,
            )
            # Map SED outputs to 234 species columns
            sed_probs_mapped = map_sed_to_species(sed_probs_raw, sed_species_mapping)
            t_sed_elapsed = time.time() - t_sed
            timing_stats["sed"].append(t_sed_elapsed)

        # ----- b. Perch inference -----
        perch_probs = None

        if perch_available:
            t_perch = time.time()

            # Load full audio at 32kHz (Perch expects 32kHz)
            audio_full, file_sr = sf.read(str(audio_path), dtype="float32")
            if audio_full.ndim == 2:
                audio_full = audio_full.mean(axis=1)
            perch_sr = 32000
            if file_sr != perch_sr:
                audio_full = librosa.resample(audio_full, orig_sr=file_sr, target_sr=perch_sr)

            # Ensure exactly 60s
            expected_samples = 60 * perch_sr
            if len(audio_full) < expected_samples:
                audio_full = np.pad(audio_full, (0, expected_samples - len(audio_full)))
            elif len(audio_full) > expected_samples:
                audio_full = audio_full[:expected_samples]

            # Split into 12 x 5s windows
            window_samples = 5 * perch_sr  # 160000
            n_windows = expected_samples // window_samples  # 12
            windows = []
            for w in range(n_windows):
                start = w * window_samples
                end = start + window_samples
                windows.append(audio_full[start:end])
            waveform_batch = np.stack(windows, axis=0).astype(np.float32)

            # Run Perch inference
            embeddings, logits = run_perch_inference(perch_infer_fn, waveform_batch)

            # Map logits to 234 BirdCLEF species
            mapped_scores = compute_perch_mapped_scores(logits, perch_species_mapping)

            # PCA transform embeddings
            pca_embeddings = pca_transformer.transform(embeddings)

            # Apply LogReg probes
            perch_probs, n_probed = apply_perch_probes(
                pca_embeddings,
                mapped_scores,
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
                alpha_event=perch_alpha_event if mode != "sed_only" else 0.15,
                alpha_texture=perch_alpha_texture if mode != "sed_only" else 0.35,
            )

            # Temperature scaling
            if perch_temperature != 1.0:
                perch_probs = _apply_temperature(perch_probs, perch_temperature)

            t_perch_elapsed = time.time() - t_perch
            timing_stats["perch"].append(t_perch_elapsed)

            # Build segment_times if SED was not available
            if segment_times is None:
                segment_times = [int((w + 1) * 5) for w in range(n_windows)]

        # ----- c. Ensemble blend -----
        t_blend = time.time()

        n_segments = len(segment_times)

        if mode == "ensemble":
            # Ensure shapes match
            if sed_probs_mapped.shape[0] != perch_probs.shape[0]:
                # Take the minimum number of segments
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
        elif mode == "sed_only":
            final_probs = sed_probs_mapped
        elif mode == "perch_only":
            final_probs = perch_probs
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        t_blend_elapsed = time.time() - t_blend
        timing_stats["blend"].append(t_blend_elapsed)

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

        # Log progress
        parts = [f"{t_file_elapsed:.1f}s total"]
        if sed_available:
            parts.append(f"SED={timing_stats['sed'][-1]:.1f}s")
        if perch_available:
            parts.append(f"Perch={timing_stats['perch'][-1]:.1f}s")
        print(
            f"  [{file_idx + 1}/{len(audio_files)}] {audio_path.name}: "
            f"{n_segments} segments ({', '.join(parts)})"
        )

    print(f"\nTotal predictions: {len(all_predictions)} rows")

    # -------------------------------------------------------------------
    # Timing summary
    # -------------------------------------------------------------------
    print("\n--- Timing Summary ---")
    for component, times in timing_stats.items():
        if times:
            print(f"  {component}: mean={np.mean(times):.2f}s, " f"total={np.sum(times):.1f}s")

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
            # No test soundscapes (public test) -- fill with zeros
            submission = sample_sub.copy()
            for col in species_columns:
                submission[col] = 0.0
        submission = submission[expected_columns]
    else:
        submission = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed_total = time.time() - start_time
    print(f"\nSubmission saved to {output_path} ({len(submission)} rows)")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"Mode: {mode} | SED weight: {sed_weight} | Perch weight: {perch_weight}")


if __name__ == "__main__":
    main()
