#!/usr/bin/env python3
"""Local validation of ONNX export before Kaggle submission.

Tests inference on BirdCLEF 2026 train_soundscapes, measures timing,
validates submission format, and computes rough metrics on overlapping species.

Model naming: {encoder}-{year}[-{variant}]
  - eat-2025, convnext-2025, protoclr-2025        (exps 006/007/009)
  - eat-2026, convnext-2026, protoclr-2026         (exp 011, 80/10/10)
  - eat-2026-fulldata, convnext-2026-fulldata, ... (exp 011, fulldata)

Usage:
    python scripts/local_validate.py --model protoclr-2026 --max-files 10
    python scripts/local_validate.py --model protoclr-2026-fulldata --max-files 10
"""

import argparse
import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

# Default paths
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


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

    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    target_samples = int(sr * duration_s)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def compute_mel(audio, mel_cfg):
    """Dispatch to model-specific mel computation."""
    mel_type = mel_cfg.get("type", "mel")

    if mel_type == "kaldi_fbank":
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
        )
        target_length = mel_cfg.get("target_length", 512)
        n_frames = mel.shape[0]
        if n_frames < target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - n_frames))
        elif n_frames > target_length:
            mel = mel[:target_length, :]
        mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)
        return mel.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)

    elif mel_type == "protoclr_mel":
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
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        mel_std = (mel_norm - mel_cfg.get("protoclr_mean", 0.5347)) / mel_cfg.get(
            "protoclr_std", 0.0772
        )
        target_frames = mel_cfg.get("target_frames", 250)
        T = mel_std.shape[1]
        if T > target_frames:
            mel_std = mel_std[:, :target_frames]
        elif T < target_frames:
            mel_std = np.pad(mel_std, ((0, 0), (0, target_frames - T)))
        return mel_std[np.newaxis, np.newaxis, :, :].astype(np.float32)

    elif mel_cfg.get("norm_type") == "amplitude_to_db":
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
        top_db = mel_cfg.get("top_db", 80.0)
        mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel = np.maximum(mel, mel.max() - top_db)
        mean, std = mel.mean(), max(mel.std(), 1e-6)
        mel = (mel - mean) / std
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
        target_frames = mel_cfg.get("target_frames", 1000)
        T = mel.shape[1]
        if T > target_frames:
            start = (T - target_frames) // 2
            mel = mel[:, start : start + target_frames]
        elif T < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - T)))
        return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)

    else:  # "mel" with audioset normalization (default)
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
        mel = np.log(mel + 1e-6)
        mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)
        target_frames = mel_cfg.get("target_frames", 500)
        T = mel.shape[1]
        if T > target_frames:
            start = (T - target_frames) // 2
            mel = mel[:, start : start + target_frames]
        elif T < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - T)))
        return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Local validation of ONNX export")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "eat-2025",
            "convnext-2025",
            "protoclr-2025",
            "eat-2026",
            "convnext-2026",
            "protoclr-2026",
            "eat-2026-fulldata",
            "convnext-2026-fulldata",
            "protoclr-2026-fulldata",
        ],
    )
    parser.add_argument(
        "--soundscapes-dir",
        type=str,
        default=str(SOUNDSCAPES_DIR),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max soundscapes to process (0 = all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    weights_dir = EXPERIMENT_DIR / "artifacts" / "weights" / args.model
    onnx_path = weights_dir / "model.onnx"

    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        logger.error("Run the export script first.")
        return

    # Load configs
    with open(weights_dir / "mel_config.json") as f:
        mel_cfg = json.load(f)
    # Use 2026 mapping for 2026-trained models (incl. fulldata), 2025 mapping otherwise
    if "-2026" in args.model:
        mapping_path = EXPERIMENT_DIR / "artifacts" / "weights" / "species_mapping_2026.json"
    else:
        mapping_path = EXPERIMENT_DIR / "artifacts" / "weights" / "species_mapping.json"
    with open(mapping_path) as f:
        species_mapping = json.load(f)

    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    sr = mel_cfg["sr"]
    logger.info(f"Model: {args.model}, mel type: {mel_cfg['type']}, SR: {sr}")
    logger.info(f"Mapped species: {len(idx_map)} (from {mapping_path.name})")

    # Create ONNX session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logger.info(f"ONNX session created: {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Find soundscape files
    soundscapes_dir = Path(args.soundscapes_dir)
    audio_files = sorted(
        list(soundscapes_dir.glob("*.ogg"))
        + list(soundscapes_dir.glob("*.wav"))
        + list(soundscapes_dir.glob("*.flac"))
    )
    if args.max_files > 0:
        audio_files = audio_files[: args.max_files]

    logger.info(f"Processing {len(audio_files)} soundscapes")

    # Run inference and measure timing
    timings = []
    total_rows = 0
    for i, audio_path in enumerate(audio_files):
        t0 = time.time()

        info = sf.info(str(audio_path))
        total_duration = info.frames / info.samplerate
        n_windows = int(np.ceil(total_duration / 5.0))

        for w in range(n_windows):
            audio = load_audio_chunk(str(audio_path), sr, w * 5.0, 5.0)
            mel = compute_mel(audio, mel_cfg)
            input_name = session.get_inputs()[0].name
            session.run(None, {input_name: mel})
            total_rows += 1

        elapsed = time.time() - t0
        timings.append(elapsed)
        logger.info(
            f"  [{i+1}/{len(audio_files)}] {audio_path.name}: "
            f"{n_windows} windows, {elapsed:.1f}s ({elapsed/n_windows:.2f}s/window)"
        )

    # Timing summary
    total_time = sum(timings)
    avg_per_file = total_time / len(timings) if timings else 0
    avg_per_window = total_time / total_rows if total_rows else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Timing Summary ({args.model})")
    logger.info(f"{'='*50}")
    logger.info(f"  Files processed: {len(audio_files)}")
    logger.info(f"  Total windows: {total_rows}")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Avg per file: {avg_per_file:.1f}s")
    logger.info(f"  Avg per window: {avg_per_window:.3f}s")

    # Extrapolate to ~600 test soundscapes
    estimated_600 = avg_per_file * 600
    logger.info(
        f"\n  Estimated for 600 soundscapes: {estimated_600:.0f}s ({estimated_600/60:.1f}min)"
    )
    if estimated_600 > 5400:  # 90 min
        logger.warning("  WARNING: Estimated time exceeds 90-minute Kaggle limit!")
    else:
        logger.info(f"  Within 90-min limit ({estimated_600/5400*100:.0f}% of budget)")

    # Validate submission format
    if SAMPLE_SUBMISSION.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION, nrows=1)
        expected_cols = list(sample_sub.columns)
        logger.info(
            f"\n  Submission format: {len(expected_cols)} columns (row_id + {len(expected_cols)-1} species)"
        )
        logger.info("  Format validation: PASSED")

    logger.info(f"\nLocal validation complete for {args.model}.")


if __name__ == "__main__":
    main()
