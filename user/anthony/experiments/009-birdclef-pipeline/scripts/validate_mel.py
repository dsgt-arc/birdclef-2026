#!/usr/bin/env python3
"""Validate librosa mel matches nnAudio mel spectrogram output.

Compares the two STFT implementations with identical parameters to ensure
the Kaggle inference script (librosa) produces compatible features.

Usage:
    python scripts/validate_mel.py --audio-path test.ogg [--sr 32000]
"""

import argparse
import logging

import librosa
import numpy as np
import soundfile as sf
import torch
from nnAudio.features import MelSpectrogram

logger = logging.getLogger(__name__)

# Mel parameters (must match finetune.yaml and export_onnx.py)
MEL_PARAMS = {
    "sr": 32000,
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 128,
    "fmin": 50,
    "fmax": 14000,
    "norm_mean": -4.268,
    "norm_std": 4.569,
}


def compute_mel_nnAudio(audio_np, params):
    """Compute mel via nnAudio (training pipeline)."""
    mel_layer = MelSpectrogram(
        sr=params["sr"],
        n_fft=params["n_fft"],
        win_length=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
        fmin=params["fmin"],
        fmax=params["fmax"],
        trainable_mel=False,
        trainable_STFT=False,
    )
    mel_layer.eval()

    waveform = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, num_samples)
    with torch.no_grad():
        mel = mel_layer(waveform)  # (1, n_mels, T)

    # Log normalize
    mel = torch.log(mel.clamp(min=1e-10))
    mel = (mel - params["norm_mean"]) / (params["norm_std"] * 2)

    return mel.squeeze(0).numpy()  # (n_mels, T)


def compute_mel_librosa(audio_np, params):
    """Compute mel via librosa (Kaggle inference pipeline)."""
    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=params["sr"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
        fmin=params["fmin"],
        fmax=params["fmax"],
        power=2.0,
    )
    # Log normalize
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = (mel - params["norm_mean"]) / (params["norm_std"] * 2)

    return mel  # (n_mels, T)


def main():
    parser = argparse.ArgumentParser(description="Validate librosa vs nnAudio mel agreement")
    parser.add_argument("--audio-path", type=str, required=True, help="Test audio file")
    parser.add_argument("--sr", type=int, default=32000, help="Target sample rate")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration to test (seconds)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = MEL_PARAMS.copy()
    params["sr"] = args.sr

    # Load audio
    audio, source_sr = sf.read(args.audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if source_sr != args.sr:
        import torchaudio
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=source_sr, new_freq=args.sr)
        audio = audio_tensor.numpy()

    # Trim to test duration
    max_samples = int(args.duration * args.sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    logger.info(f"Audio: {args.audio_path} ({len(audio)/args.sr:.1f}s, {len(audio)} samples)")

    # Compute mels
    mel_nn = compute_mel_nnAudio(audio, params)
    mel_lr = compute_mel_librosa(audio, params)

    logger.info(f"nnAudio mel shape: {mel_nn.shape}")
    logger.info(f"librosa mel shape: {mel_lr.shape}")

    # Align shapes (may differ by 1 frame due to padding)
    min_t = min(mel_nn.shape[1], mel_lr.shape[1])
    mel_nn = mel_nn[:, :min_t]
    mel_lr = mel_lr[:, :min_t]

    # Compare
    diff = np.abs(mel_nn - mel_lr)
    max_diff = diff.max()
    mean_diff = diff.mean()
    p99_diff = np.percentile(diff, 99)

    logger.info(f"Absolute difference statistics:")
    logger.info(f"  Max:  {max_diff:.6f}")
    logger.info(f"  Mean: {mean_diff:.6f}")
    logger.info(f"  P99:  {p99_diff:.6f}")

    # Per-mel-bin stats
    per_bin_max = diff.max(axis=1)
    logger.info(f"  Per-bin max range: [{per_bin_max.min():.6f}, {per_bin_max.max():.6f}]")

    # Correlation
    corr = np.corrcoef(mel_nn.flatten(), mel_lr.flatten())[0, 1]
    logger.info(f"  Pearson correlation: {corr:.6f}")

    # Verdict
    threshold = 0.01
    if max_diff < threshold:
        logger.info(f"PASS: max diff {max_diff:.6f} < {threshold}")
    else:
        logger.warning(f"FAIL: max diff {max_diff:.6f} >= {threshold}")
        logger.warning("Consider adjusting librosa parameters or using a tolerance in inference")

    return max_diff < threshold


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
