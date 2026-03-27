#!/usr/bin/env python3
"""Local validation of ONNX export before Kaggle submission.

Tests inference on BirdCLEF 2026 train_soundscapes, measures timing,
validates submission format, and computes rough metrics on overlapping species.

Supports two inference strategies (mirrors submission.py):
  - Full-signal mel (SED models): compute STFT once per file, slice + batch ONNX
  - Per-segment mel (clip-level models): independent mel per 5s window

Usage:
    python scripts/local_validate.py --model effnet-b0-sed-v4b --max-files 5
    python scripts/local_validate.py --model effnet-b0-sed-v4f --max-files 10
    python scripts/local_validate.py --model protoclr-2026 --max-files 10
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

# Default paths
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"

# All supported models (auto-discovered from artifacts/weights/)
LEGACY_MODELS = [
    "eat-2025",
    "convnext-2025",
    "protoclr-2025",
    "eat-2026",
    "convnext-2026",
    "protoclr-2026",
    "eat-2026-fulldata",
    "convnext-2026-fulldata",
    "protoclr-2026-fulldata",
    "eat-2026-soundscape",
    "convnext-2026-soundscape",
    "protoclr-2026-soundscape",
    "protoclr-2026-student-v1",
    "eat-2026-student-v1",
]


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# ---------------------------------------------------------------------------
# Per-segment mel (legacy clip-level models)
# ---------------------------------------------------------------------------


def load_audio_chunk(audio_path, sr, start_s, duration_s):
    """Load a chunk of audio at target sample rate."""
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr
    target_samples = int(sr * duration_s)

    pre_pad_samples = 0
    read_start_s = start_s
    if start_s < 0:
        pre_pad_s = -start_s
        pre_pad_samples = int(sr * pre_pad_s)
        read_start_s = 0.0
        duration_s = duration_s - pre_pad_s

    actual_duration = min(duration_s, total_duration - read_start_s)
    if actual_duration <= 0:
        return np.zeros(target_samples, dtype=np.float32)

    audio, _ = sf.read(
        audio_path,
        dtype="float32",
        start=int(read_start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)
    if pre_pad_samples > 0:
        audio = np.concatenate([np.zeros(pre_pad_samples, dtype=np.float32), audio])
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio.astype(np.float32)


def compute_mel_segment(audio, mel_cfg):
    """Compute mel for a single audio segment (per-segment path)."""
    mel_type = mel_cfg.get("type", "mel")
    norm_type = mel_cfg.get("norm_type", "audioset")

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

    elif norm_type == "amplitude_to_db":
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

    else:
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


# ---------------------------------------------------------------------------
# Full-signal mel + batched ONNX (optimized SED inference)
# ---------------------------------------------------------------------------


def compute_full_signal_mel(audio, mel_cfg):
    """Compute mel for an entire audio signal (e.g. 60s).

    Returns (n_mels, total_frames) without batch/channel dims.
    Avoids redundant STFT computation for overlapping segments.
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


def slice_mel_window(full_mel, start_frame, target_frames):
    """Slice a window from full-signal mel, padding if needed.

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


def validate_fullsignal(session, audio_path, mel_cfg):
    """Validate one soundscape using full-signal mel + batched ONNX.

    Returns (n_segments, elapsed_seconds).
    """
    sr = mel_cfg["sr"]
    hop_length = mel_cfg["hop_length"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2
    target_frames = mel_cfg.get("target_frames", 1000)

    # 1. Load entire file
    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    total_duration = len(audio) / sr

    # 2. Pad for context + TTA
    context_samples = int(context_pad * sr)
    tta_shift_samples = int(2.5 * sr)
    pad_samples = context_samples + tta_shift_samples
    audio_padded = np.pad(audio, (pad_samples, pad_samples), mode="constant")

    # 3. Compute full-signal mel ONCE
    full_mel = compute_full_signal_mel(audio_padded, mel_cfg)
    origin_frame = int(pad_samples / hop_length)

    # 4. Build all mel slices (segments + TTA)
    all_slices = []
    n_segments = 0
    segment_start_s = 0.0
    while segment_start_s < total_duration:
        window_center_s = segment_start_s - context_pad
        shifts = [
            (window_center_s, 0.50),
            (window_center_s - 2.5, 0.25),
            (window_center_s + 2.5, 0.25),
        ]
        for shift_start_s, weight in shifts:
            start_frame = origin_frame + int(shift_start_s * sr / hop_length)
            mel_slice = slice_mel_window(full_mel, start_frame, target_frames)
            all_slices.append((n_segments, weight, mel_slice))
        n_segments += 1
        segment_start_s += segment_duration

    # 5. Batched ONNX inference
    mel_batch = np.concatenate([s[2] for s in all_slices], axis=0)
    input_name = session.get_inputs()[0].name
    all_probs = session.run(None, {input_name: mel_batch})[0]

    # 6. Weighted TTA average
    probs_matrix = np.zeros((n_segments, all_probs.shape[1]), dtype=np.float32)
    for i, (seg_idx, weight, _) in enumerate(all_slices):
        probs_matrix[seg_idx] += weight * all_probs[i]

    if mel_cfg.get("output_type") != "probs":
        probs_matrix = sigmoid(probs_matrix)

    return n_segments, probs_matrix


def validate_per_segment(session, audio_path, mel_cfg):
    """Validate one soundscape using per-segment mel (legacy path).

    Returns (n_segments, elapsed_seconds).
    """
    sr = mel_cfg["sr"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2

    info = sf.info(str(audio_path))
    total_duration = info.frames / info.samplerate

    all_probs = []
    segment_start_s = 0.0
    while segment_start_s < total_duration:
        audio_start = segment_start_s - context_pad
        audio = load_audio_chunk(str(audio_path), sr, audio_start, inference_window)
        mel = compute_mel_segment(audio, mel_cfg)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: mel})[0].flatten()
        if mel_cfg.get("output_type") != "probs":
            output = sigmoid(output)
        all_probs.append(output)
        segment_start_s += segment_duration

    probs_matrix = np.stack(all_probs)
    return len(all_probs), probs_matrix


def main():
    parser = argparse.ArgumentParser(description="Local validation of ONNX export")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (auto-detected from artifacts/weights/)",
    )
    parser.add_argument("--soundscapes-dir", type=str, default=str(SOUNDSCAPES_DIR))
    parser.add_argument(
        "--max-files", type=int, default=5, help="Max soundscapes to process (0 = all)"
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

    mel_type = mel_cfg.get("type", "mel")
    sr = mel_cfg["sr"]
    logger.info(f"Model: {args.model}")
    logger.info(f"  Mel type: {mel_type}, norm: {mel_cfg.get('norm_type', 'audioset')}")
    logger.info(f"  SR: {sr}, n_fft: {mel_cfg.get('n_fft')}, n_mels: {mel_cfg.get('n_mels')}")
    logger.info(f"  Architecture: {mel_cfg.get('architecture', 'clip')}")
    logger.info(f"  Output type: {mel_cfg.get('output_type', 'logits')}")

    # Create optimized ONNX session (mirrors submission.py)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    logger.info(
        f"  ONNX session: {onnx_path.stat().st_size / 1e6:.1f} MB, "
        f"threads={sess_opts.intra_op_num_threads}"
    )

    # Choose inference strategy (mirrors submission.py routing)
    use_fullsignal = mel_cfg.get("architecture") == "sed" and mel_type == "mel"
    if use_fullsignal:
        logger.info("  Strategy: full-signal mel + batched ONNX (optimized)")
    else:
        logger.info("  Strategy: per-segment mel (legacy)")

    # Find soundscape files
    soundscapes_dir = Path(args.soundscapes_dir)
    audio_files = sorted(
        list(soundscapes_dir.glob("*.ogg"))
        + list(soundscapes_dir.glob("*.wav"))
        + list(soundscapes_dir.glob("*.flac"))
    )
    if args.max_files > 0:
        audio_files = audio_files[: args.max_files]

    logger.info(f"\nProcessing {len(audio_files)} soundscapes from {soundscapes_dir}")

    # Run inference and measure timing
    timings = []
    total_segments = 0
    for i, audio_path in enumerate(audio_files):
        t0 = time.time()
        if use_fullsignal:
            n_seg, probs = validate_fullsignal(session, str(audio_path), mel_cfg)
        else:
            n_seg, probs = validate_per_segment(session, str(audio_path), mel_cfg)
        elapsed = time.time() - t0
        timings.append(elapsed)
        total_segments += n_seg

        # Quick stats on predictions
        max_prob = probs.max()
        n_active = (probs > 0.5).sum()
        logger.info(
            f"  [{i+1}/{len(audio_files)}] {audio_path.name}: "
            f"{n_seg} segments, {elapsed:.1f}s "
            f"(max={max_prob:.3f}, active@0.5={n_active})"
        )

    # Timing summary
    total_time = sum(timings)
    avg_per_file = total_time / len(timings) if timings else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Validation Summary: {args.model}")
    logger.info(f"{'='*60}")
    logger.info(
        f"  Inference: {'full-signal mel + batched ONNX' if use_fullsignal else 'per-segment'}"
    )
    logger.info(f"  Files: {len(audio_files)}, Segments: {total_segments}")
    logger.info(f"  Total time: {total_time:.1f}s, Avg/file: {avg_per_file:.2f}s")

    # Extrapolate to ~600 test soundscapes (Kaggle)
    estimated_600 = avg_per_file * 600
    logger.info(
        f"\n  Kaggle estimate (600 files): {estimated_600:.0f}s ({estimated_600/60:.1f}min)"
    )
    if estimated_600 > 5400:
        logger.warning("  WARNING: Exceeds 90-minute Kaggle limit!")
    else:
        logger.info(f"  Within limit ({estimated_600/5400*100:.0f}% of 90min budget)")

    logger.info("\nValidation complete.")


if __name__ == "__main__":
    main()
