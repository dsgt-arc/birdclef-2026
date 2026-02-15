#!/usr/bin/env python3
"""Self-contained Kaggle inference script for BirdCLEF 2025.

Runs offline on Kaggle CPU kernel (no PyTorch, no internet).
Dependencies are installed from pre-uploaded wheel datasets.

Pipeline per soundscape:
  for each 10s chunk:
    1. Load audio at 32kHz via soundfile
    2. Compute mel via librosa (matching nnAudio parameters)
    3. Log-normalize: (log(mel.clip(1e-10)) - mean) / (std * 2)
    4. Pad/trim to 1000 frames -> (1, 1, 128, 1000)
    5. ONNX session.run() -> (1, 1000, C) per-token logits
    6. Split into 2x500-token windows, mean pool each -> (2, C)
    7. sigmoid -> probs, write two 5s rows

The full 1000 tokens are processed in one pass so Mamba's hidden state
propagates from the first 5s window into the second.
"""

import subprocess
import sys

# ---------------------------------------------------------------------------
# Offline dependency installation (runs on Kaggle before imports)
# ---------------------------------------------------------------------------
KAGGLE_USER = "acmiyaguchi"

# Support both old (/kaggle/input/<slug>) and new (/kaggle/input/datasets/<user>/<slug>) layouts
import os

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

DEPS_DIR = _find_input("birdclef-2025-deps")
WEIGHTS_DIR = _find_input("birdclef-2025-weights")
COMPETITION_DIR = (
    "/kaggle/input/competitions/birdclef-2025"
    if os.path.exists("/kaggle/input/competitions/birdclef-2025")
    else "/kaggle/input/birdclef-2025"
)
if os.path.exists(DEPS_DIR):
    print(f"Installing deps from {DEPS_DIR}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", DEPS_DIR,
        "onnxruntime", "mamba_onnx", "librosa", "soundfile", "numpy", "pandas",
    ])

# ---------------------------------------------------------------------------
# Imports (available after offline install)
# ---------------------------------------------------------------------------
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import onnxruntime as ort


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_mel(audio, mel_cfg):
    """Compute normalized log-mel spectrogram matching training pipeline.

    Args:
        audio: (num_samples,) float32 array at mel_cfg["sr"]
        mel_cfg: dict with sr, n_fft, hop_length, n_mels, fmin, fmax, norm_mean, norm_std

    Returns:
        (1, 1, n_mels, target_frames) float32 array, padded/trimmed to 1000 frames
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
        power=2.0,
    )
    # Log normalize (matching nnAudio pipeline)
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    # Pad or trim to target_frames (1000)
    target_frames = 1000
    T = mel.shape[1]
    if T > target_frames:
        mel = mel[:, :target_frames]
    elif T < target_frames:
        pad_width = target_frames - T
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)

    # Shape: (1, 1, n_mels, target_frames)
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def load_audio_chunk(audio_path, sr, start_s, duration_s):
    """Load a chunk of audio at target sample rate.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        start_s: Start time in seconds
        duration_s: Duration in seconds

    Returns:
        (num_samples,) float32 array
    """
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr

    # Clamp to file length
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


def predict_soundscape(session, audio_path, species_list, mel_cfg, window_size=500):
    """Process one soundscape file and return prediction rows.

    Each 10s chunk produces two 5s rows via windowed mean-pooling.

    Args:
        session: ONNX InferenceSession
        audio_path: Path to soundscape .ogg file
        species_list: Ordered list of species names
        mel_cfg: Mel config dict
        window_size: Tokens per window (500 = 5s)

    Returns:
        List of dicts with row_id and per-species probabilities
    """
    sr = mel_cfg["sr"]
    chunk_duration = 10.0
    soundscape_id = Path(audio_path).stem

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    predictions = []
    chunk_start_s = 0.0

    while chunk_start_s < total_duration:
        # Load 10s chunk
        audio = load_audio_chunk(audio_path, sr, chunk_start_s, chunk_duration)

        # Compute mel: (1, 1, 128, 1000)
        mel = compute_mel(audio, mel_cfg)

        # ONNX inference: (1, 1000, C)
        input_name = session.get_inputs()[0].name
        token_logits = session.run(None, {input_name: mel})[0]  # (1, T, C)

        T = token_logits.shape[1]
        num_windows = T // window_size

        for w in range(num_windows):
            # Mean pool this window's tokens
            start_tok = w * window_size
            end_tok = start_tok + window_size
            window_logits = token_logits[0, start_tok:end_tok, :].mean(axis=0)  # (C,)

            # Sigmoid to probabilities
            probs = sigmoid(window_logits)

            end_time = int(chunk_start_s + (w + 1) * 5)
            row_id = f"{soundscape_id}_{end_time}"
            row = {"row_id": row_id}
            for i, species in enumerate(species_list):
                row[species] = float(probs[i])
            predictions.append(row)

        chunk_start_s += chunk_duration

    return predictions


def main():
    start_time = time.time()

    # Paths (Kaggle layout)
    weights_dir = Path(WEIGHTS_DIR) if os.path.exists(WEIGHTS_DIR) else Path("artifacts/weights")
    test_dir = Path(COMPETITION_DIR) / "test_soundscapes"
    sample_sub_path = Path(COMPETITION_DIR) / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")

    # Fall back to local paths for testing
    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not sample_sub_path.exists():
        sample_sub_path = Path(os.environ.get("SAMPLE_SUBMISSION", "sample_submission.csv"))
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    # Load model artifacts
    onnx_path = weights_dir / "ssamba_classifier.onnx"
    species_path = weights_dir / "species_list.json"
    mel_config_path = weights_dir / "mel_config.json"

    print(f"Loading model from {onnx_path}")
    with open(species_path) as f:
        species_list = json.load(f)
    with open(mel_config_path) as f:
        mel_cfg = json.load(f)

    print(f"Species: {len(species_list)}, Mel config: {mel_cfg}")

    # Create ONNX session with mamba-onnx custom ops
    session_options = ort.SessionOptions()
    try:
        import mamba_onnx
        lib_path = mamba_onnx.register_custom_ops(session_options, backend="zig")
        print(f"mamba-onnx custom ops loaded from {lib_path}")
    except ImportError:
        print("WARNING: mamba-onnx not available, custom ops may fail")

    # CPU-only inference
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    print(f"ONNX session created ({time.time() - start_time:.1f}s)")

    # Find soundscape files
    audio_files = sorted(
        list(test_dir.glob("*.ogg"))
        + list(test_dir.glob("*.wav"))
        + list(test_dir.glob("*.flac"))
    )
    print(f"Processing {len(audio_files)} soundscapes from {test_dir}")

    # Run inference
    all_predictions = []
    for i, audio_path in enumerate(audio_files):
        t0 = time.time()
        preds = predict_soundscape(session, str(audio_path), species_list, mel_cfg)
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}: {len(preds)} rows ({elapsed:.1f}s)")
        all_predictions.extend(preds)

    print(f"Total predictions: {len(all_predictions)} rows")

    # Build submission
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
