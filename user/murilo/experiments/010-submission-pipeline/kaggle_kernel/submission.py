#!/usr/bin/env python3
"""Unified BirdCLEF 2026 Kaggle inference script.

Self-contained Kaggle CPU kernel for BirdCLEF 2026 submission.
Auto-detects which model (EAT, ConvNeXt, ProtoCLR) is available in the
weights directory and runs the appropriate preprocessing pipeline.

Supports models trained on either BirdCLEF 2025 or 2026 data:
  - 2025-trained: 41/206 model outputs map to 2026 columns (rest get 0.0)
  - 2026-trained: 206/206 model outputs map to 2026 columns

Pipeline per soundscape:
  for each 5s window:
    1. Load audio at model-specific sample rate
    2. Compute mel spectrogram (model-specific preprocessing)
    3. ONNX session.run() -> logits
    4. sigmoid -> probs
    5. Map model probs to 234 BirdCLEF-2026 columns via species_mapping.json
    6. Unmapped species get 0.0

Supports three mel types:
  - kaldi_fbank (EAT): torchaudio Kaldi fbank features
  - mel (ConvNeXt): librosa log-mel spectrogram + AudioSet normalization
  - protoclr_mel (ProtoCLR): librosa mel + dB + min-max + ProtoCLR standardization
"""

# ---------------------------------------------------------------------------
# Offline dependency installation (runs on Kaggle before imports)
# Wheels are in a separate Kaggle dataset: birdclef-deps
# ---------------------------------------------------------------------------
import os
import subprocess
import sys

KAGGLE_USER = "murilogustineli"


def _find_input(slug: str) -> str:
    """Resolve Kaggle input path, supporting both old and new directory layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # fallback to old-style


DEPS_DIR = _find_input("birdclef-deps")

# Debug: show what we see in /kaggle/input
if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

# ---------------------------------------------------------------------------
# Kaggle path resolution
# ---------------------------------------------------------------------------
KAGGLE_USER = "murilogustineli"

# Model slug names — the kernel metadata picks one of these as a dataset source
MODEL_SLUGS = [
    # SED models, 234 classes (exp 016)
    "birdclef-2026-effnet-b0-sed-v1-weights",
    # Student models trained on focal + soundscape + pseudo-labels, 234 classes (exp 015)
    "birdclef-2026-protoclr-student-v1-weights",
    "birdclef-2026-eat-student-v1-weights",
    # Models fine-tuned on focal + labeled soundscapes, 234 classes (exp 012)
    "birdclef-2026-protoclr-soundscape-weights",
    "birdclef-2026-eat-soundscape-weights",
    "birdclef-2026-convnext-soundscape-weights",
    # Models trained on BirdCLEF 2026 focal data, 100% fulldata (206/234 species mapped)
    "birdclef-2026-protoclr-fulldata-weights",
    "birdclef-2026-eat-fulldata-weights",
    "birdclef-2026-convnext-fulldata-weights",
    # Models trained on BirdCLEF 2026 focal data, 80/10/10 split (206/234 species mapped)
    "birdclef-2026-protoclr-weights",
    "birdclef-2026-eat-weights",
    "birdclef-2026-convnext-weights",
    # Models trained on BirdCLEF 2025 data (41/234 species mapped)
    "birdclef-2025-eat-weights",
    "birdclef-2025-convnext-weights",
    "birdclef-2025-protoclr-weights",
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


def find_weights_dir() -> Path:
    """Auto-detect which model weights are available."""
    for slug in MODEL_SLUGS:
        path = _find_input(slug)
        if path and os.path.exists(path):
            return Path(path)

    # Local testing fallback
    local = Path("artifacts/weights")
    for subdir in [
        "protoclr-2026-soundscape",
        "eat-2026-soundscape",
        "convnext-2026-soundscape",
        "protoclr-2026-fulldata",
        "eat-2026-fulldata",
        "convnext-2026-fulldata",
        "protoclr-2026",
        "eat-2026",
        "convnext-2026",
        "eat-2025",
        "convnext-2025",
        "protoclr-2025",
    ]:
        if (local / subdir / "model.onnx").exists():
            return local / subdir
    if local.exists():
        return local

    raise FileNotFoundError("No model weights found")


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


# ---------------------------------------------------------------------------
# Mel spectrogram computation
# ---------------------------------------------------------------------------


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_mel_standard(audio, mel_cfg):
    """Compute log-mel spectrogram with AudioSet normalization (ConvNeXt).

    Matches NNAudioMelSpectrogram: log(mel + 1e-6), normalize by AudioSet stats.
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
    # Log scale (matching nnAudio: additive epsilon)
    mel = np.log(mel + 1e-6)

    # Normalize: (mel - mean) / (std * 2)
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 500)
    T = mel.shape[1]
    if T > target_frames:
        # Center crop
        start = (T - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, 128, 500)


def compute_mel_kaldi_fbank(audio, mel_cfg):
    """Compute Kaldi fbank features for EAT.

    Uses torchaudio.compliance.kaldi.fbank (available on Kaggle).
    """
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float()
    waveform = waveform - waveform.mean()

    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=mel_cfg["sr"],
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_cfg["num_mel_bins"],
        dither=0.0,
        frame_shift=mel_cfg["frame_shift"],
    )  # (time, 128)

    # Pad/truncate to target length
    target_length = mel_cfg.get("target_length", 512)
    n_frames = mel.shape[0]
    if n_frames < target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - n_frames))
    elif n_frames > target_length:
        mel = mel[:target_length, :]

    # Normalize
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    # Shape: (1, 1, target_length, 128)
    return mel.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)


def compute_mel_protoclr(audio, mel_cfg):
    """Compute ProtoCLR-compatible mel spectrogram.

    Pipeline: mel spectrogram -> dB scale -> min-max norm -> standardize.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 50),
        fmax=mel_cfg.get("fmax", 8000),
        power=2.0,
    )

    # Convert to dB (matches torchaudio.transforms.AmplitudeToDB default: ref=1.0, power)
    # AmplitudeToDB with stype="power": 10 * log10(mel / ref)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))

    # Per-sample min-max normalization
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # Standardize with ProtoCLR training set statistics
    protoclr_mean = mel_cfg.get("protoclr_mean", 0.5347)
    protoclr_std = mel_cfg.get("protoclr_std", 0.0772)
    mel_std = (mel_norm - protoclr_mean) / protoclr_std

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 250)
    T = mel_std.shape[1]
    if T > target_frames:
        mel_std = mel_std[:, :target_frames]
    elif T < target_frames:
        mel_std = np.pad(mel_std, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel_std[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, 128, 250)


def compute_mel(audio, mel_cfg):
    """Dispatch to model-specific mel computation."""
    mel_type = mel_cfg.get("type", "mel")
    if mel_type == "kaldi_fbank":
        return compute_mel_kaldi_fbank(audio, mel_cfg)
    elif mel_type == "protoclr_mel":
        return compute_mel_protoclr(audio, mel_cfg)
    else:
        return compute_mel_standard(audio, mel_cfg)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_chunk(audio_path, sr, start_s, duration_s):
    """Load a chunk of audio at target sample rate."""
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr

    actual_duration = min(duration_s, total_duration - start_s)
    if actual_duration <= 0:
        return np.zeros(int(sr * duration_s), dtype=np.float32)

    audio, _ = sf.read(
        audio_path,
        dtype="float32",
        start=int(start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    # Pad to full chunk length
    target_samples = int(sr * duration_s)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _run_onnx(session, mel, mel_cfg):
    """Run ONNX inference and return probability vector."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel})[0]
    output = output.flatten()

    # SED models output blended probabilities directly; clip-level models output logits
    if mel_cfg.get("output_type") == "probs":
        return output
    else:
        return sigmoid(output)


def _predict_segment_with_tta(
    session, audio_path, sr, window_start_s, window_duration, mel_cfg, use_tta=True
):
    """Predict one segment, optionally with temporal TTA (2.5s shifts).

    TTA: run 3 predictions per segment (center, -2.5s, +2.5s) and average.
    Confirmed +0.012 LB from competition analysis.
    """
    audio = load_audio_chunk(audio_path, sr, window_start_s, window_duration)
    mel = compute_mel(audio, mel_cfg)
    probs = _run_onnx(session, mel, mel_cfg)

    if not use_tta:
        return probs

    # TTA: shifted predictions
    shift_s = window_duration / 2  # 2.5s for 5s windows
    tta_preds = [probs]

    for shift in [-shift_s, shift_s]:
        shifted_start = window_start_s + shift
        if shifted_start < 0:
            shifted_start = 0
        audio_shifted = load_audio_chunk(audio_path, sr, shifted_start, window_duration)
        mel_shifted = compute_mel(audio_shifted, mel_cfg)
        tta_preds.append(_run_onnx(session, mel_shifted, mel_cfg))

    # Average: 50% center + 25% left + 25% right
    return 0.5 * tta_preds[0] + 0.25 * tta_preds[1] + 0.25 * tta_preds[2]


def _apply_temporal_smoothing(probs_matrix):
    """Gaussian temporal smoothing across consecutive segments.

    Kernel [0.1, 0.2, 0.4, 0.2, 0.1] — confirmed +0.010 LB.
    """
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    n_segments, n_species = probs_matrix.shape
    if n_segments < 3:
        return probs_matrix

    smoothed = np.copy(probs_matrix)
    for c in range(n_species):
        smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
    return smoothed


def _apply_topn(probs_matrix):
    """TopN postprocessing: multiply each segment by file-level max per species.

    Boosts species that are consistently detected across the file and suppresses
    spurious single-segment detections. Confirmed +0.011 LB.
    """
    file_max = probs_matrix.max(axis=0, keepdims=True)
    return probs_matrix * file_max


# Post-processing flags (set to True to enable)
ENABLE_TTA = True
ENABLE_TEMPORAL_SMOOTHING = True
ENABLE_TOPN = True


def predict_soundscape(session, audio_path, mel_cfg, species_mapping):
    """Process one soundscape file and return prediction rows.

    Each 5s window produces one row with 234 BirdCLEF 2026 species columns.
    Applies post-processing: TTA → temporal smoothing → TopN.
    """
    sr = mel_cfg["sr"]
    window_duration = 5.0
    soundscape_id = Path(audio_path).stem

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]

    # Collect all segment probabilities as a matrix for post-processing
    segment_probs = []
    segment_times = []
    window_start_s = 0.0

    while window_start_s < total_duration:
        probs = _predict_segment_with_tta(
            session,
            audio_path,
            sr,
            window_start_s,
            window_duration,
            mel_cfg,
            use_tta=ENABLE_TTA,
        )
        segment_probs.append(probs)
        segment_times.append(int(window_start_s + window_duration))
        window_start_s += window_duration

    # Stack into (n_segments, n_model_outputs) matrix
    probs_matrix = np.stack(segment_probs)  # (n_segments, n_model_outputs)

    # Apply post-processing pipeline
    if ENABLE_TEMPORAL_SMOOTHING:
        probs_matrix = _apply_temporal_smoothing(probs_matrix)

    if ENABLE_TOPN:
        probs_matrix = _apply_topn(probs_matrix)

    # Build prediction rows mapped to 234 submission columns
    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}

        for sp in species_2026:
            row[sp] = 0.0

        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(probs_matrix[i, int(idx_str)])

        predictions.append(row)

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()

    # Resolve paths
    weights_dir = find_weights_dir()
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

    # Load model artifacts
    onnx_path = weights_dir / "model.onnx"
    mel_config_path = weights_dir / "mel_config.json"
    species_mapping_path = weights_dir / "species_mapping.json"

    print(f"Model weights: {weights_dir}")
    print(f"ONNX model: {onnx_path}")

    with open(mel_config_path) as f:
        mel_cfg = json.load(f)
    with open(species_mapping_path) as f:
        species_mapping = json.load(f)

    mel_type = mel_cfg.get("type", "mel")
    print(f"Mel type: {mel_type}, SR: {mel_cfg['sr']}")
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    print(f"Species mapping: {len(idx_map)} mapped to 2026 columns")

    # Create ONNX session (CPU only)
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    print(f"ONNX session created ({time.time() - start_time:.1f}s)")

    # Find soundscape files
    audio_files = sorted(
        list(test_dir.glob("*.ogg")) + list(test_dir.glob("*.wav")) + list(test_dir.glob("*.flac"))
    )

    # Fallback: if test_soundscapes is empty (public run), use train_soundscapes
    # so we can validate the full inference pipeline in visible logs
    if len(audio_files) == 0:
        train_dir = competition_dir / "train_soundscapes"
        if train_dir.exists():
            audio_files = sorted(train_dir.glob("*.ogg"))[:2]
            print(f"No test soundscapes found, using {len(audio_files)} train files for validation")

    print(f"Processing {len(audio_files)} soundscapes from {test_dir}")

    # Run inference
    all_predictions = []
    for i, audio_path in enumerate(audio_files):
        t0 = time.time()
        preds = predict_soundscape(session, str(audio_path), mel_cfg, species_mapping)
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}: {len(preds)} rows ({elapsed:.1f}s)")
        all_predictions.extend(preds)

    print(f"Total predictions: {len(all_predictions)} rows")

    # Build submission merged with sample_submission.csv
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
            # No test soundscapes (public test) — fill with zeros
            submission = sample_sub.copy()
            for col in species_columns:
                submission[col] = 0.0
        submission = submission[expected_columns]
    else:
        submission = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed_total = time.time() - start_time
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")


if __name__ == "__main__":
    main()
