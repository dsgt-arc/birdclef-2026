#!/usr/bin/env python3
"""Kaggle inference script for BirdCLEF 2026 (EfficientNet-B1).

CPU-only, offline. Dependencies installed from pre-uploaded wheel dataset.

Pipeline per soundscape:
  for each 5s window:
    1. Load audio at 32kHz via soundfile
    2. Compute mel via librosa (n_mels=256, n_fft=2048, hop=256)
    3. Log-normalize: (log(mel.clip(1e-10)) - mean) / (std * 2)
    4. Pad/trim to 625 frames -> (1, 1, 256, 625)
    5. ONNX session.run() -> (1, C) clip logits
    6. sigmoid -> probs
"""

import os
import subprocess
import sys
import time

# --- Offline dependency installation ---
KAGGLE_USER = "acmiyaguchi"


def _find_input(slug: str) -> str:
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# --- Model selection ---
# Change this to switch between models:
#   "frozen"              - Run A: EfficientNet frozen, epoch 25, cmap 0.387
#   "lora"                - Run B: EfficientNet LoRA, epoch 1, cmap 0.524
#   "lora-attention"      - Run C: EfficientNet LoRA+attention, epoch 1, cmap 0.531
#   "convnext-avg"        - Run D: ConvNeXt LoRA avg, epoch 8, cmap 0.644
#   "convnext-attention"  - Run E: ConvNeXt LoRA attention, epoch 5, cmap 0.585
#   "convnext-soundscape" - Run F: ConvNeXt LoRA attention+SC, epoch 5, cmap 0.591
MODEL_NAME = "lora-attention"  # Best LB: 0.794

DEPS_DIR = _find_input("birdclef-2026-deps")
_weights_base = _find_input("birdclef-2026-efficientnet-weights")
WEIGHTS_DIR = os.path.join(_weights_base, MODEL_NAME)
COMPETITION_DIR = (
    "/kaggle/input/competitions/birdclef-2026"
    if os.path.exists("/kaggle/input/competitions/birdclef-2026")
    else "/kaggle/input/birdclef-2026"
)

if os.path.exists(DEPS_DIR):
    print(f"Installing deps from {DEPS_DIR}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", DEPS_DIR,
        "onnxruntime", "librosa", "soundfile", "numpy", "pandas",
    ])

# --- Imports ---
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import onnxruntime as ort


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_mel(audio, mel_cfg):
    """Compute normalized log-mel matching training pipeline."""
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
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    target_frames = mel_cfg.get("target_frames", 625)
    T = mel.shape[1]
    if T > target_frames:
        mel = mel[:, :target_frames]
    elif T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def load_audio_chunk(audio_path, sr, start_s, duration_s):
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr

    actual_duration = min(duration_s, total_duration - start_s)
    if actual_duration <= 0:
        return np.zeros(int(sr * duration_s), dtype=np.float32)

    audio, _ = sf.read(
        audio_path, dtype="float32",
        start=int(start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    target_samples = int(sr * duration_s)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def predict_soundscape(session, audio_path, species_list, mel_cfg):
    sr = mel_cfg["sr"]
    window_duration = 5.0
    soundscape_id = Path(audio_path).stem

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    predictions = []
    window_start_s = 0.0

    while window_start_s < total_duration:
        audio = load_audio_chunk(audio_path, sr, window_start_s, window_duration)
        mel = compute_mel(audio, mel_cfg)

        input_name = session.get_inputs()[0].name
        clip_logits = session.run(None, {input_name: mel})[0]
        # Skip sigmoid if model already outputs probabilities (attention pooling)
        if mel_cfg.get("output_is_probs", False):
            probs = clip_logits[0]
        else:
            probs = sigmoid(clip_logits[0])

        end_time = int(window_start_s + window_duration)
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for i, species in enumerate(species_list):
            row[species] = float(probs[i])
        predictions.append(row)

        window_start_s += window_duration

    return predictions


def main():
    start_time = time.time()

    weights_dir = Path(WEIGHTS_DIR) if os.path.exists(WEIGHTS_DIR) else Path("artifacts/weights")
    test_dir = Path(COMPETITION_DIR) / "test_soundscapes"
    sample_sub_path = Path(COMPETITION_DIR) / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")

    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not sample_sub_path.exists():
        sample_sub_path = Path(os.environ.get("SAMPLE_SUBMISSION", "sample_submission.csv"))
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    onnx_path = weights_dir / "model.onnx"
    with open(weights_dir / "species_list.json") as f:
        species_list = json.load(f)
    with open(weights_dir / "mel_config.json") as f:
        mel_cfg = json.load(f)

    print(f"Species: {len(species_list)}, Mel: {mel_cfg}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"ONNX session ready ({time.time() - start_time:.1f}s)")

    audio_files = sorted(
        list(test_dir.glob("*.ogg")) + list(test_dir.glob("*.wav")) + list(test_dir.glob("*.flac"))
    )
    print(f"Processing {len(audio_files)} soundscapes")

    all_predictions = []
    for i, audio_path in enumerate(audio_files):
        t0 = time.time()
        preds = predict_soundscape(session, str(audio_path), species_list, mel_cfg)
        print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}: {len(preds)} rows ({time.time()-t0:.1f}s)")
        all_predictions.extend(preds)

    print(f"Total: {len(all_predictions)} rows")

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

    print(f"Saved to {output_path} ({len(submission)} rows, {time.time()-start_time:.1f}s)")


if __name__ == "__main__":
    main()
