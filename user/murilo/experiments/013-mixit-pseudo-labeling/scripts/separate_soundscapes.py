#!/usr/bin/env python3
"""Batch source separation of BirdCLEF 2026 soundscapes using Bird MixIT (ONNX).

Runs the pretrained Bird MixIT 4-source separation model on unlabeled soundscape
files, saving 4 separated channels per file as compressed numpy arrays.

The model expects 22.05 kHz mono input and outputs 4 separated waveforms that
sum exactly to the input (mixture consistency).

Usage:
    # Separate all unlabeled soundscapes
    python scripts/separate_soundscapes.py \
        --model /storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx \
        --soundscape-dir /storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes \
        --output-dir artifacts/separated \
        --exclude-labeled

    # Separate a small subset for testing
    python scripts/separate_soundscapes.py \
        --model /path/to/bird_mixit_4sources.onnx \
        --soundscape-dir /path/to/train_soundscapes \
        --output-dir artifacts/separated \
        --max-files 10

    # Separate labeled soundscapes only (for validation)
    python scripts/separate_soundscapes.py \
        --model /path/to/bird_mixit_4sources.onnx \
        --soundscape-dir /path/to/train_soundscapes \
        --output-dir artifacts/separated_labeled \
        --only-labeled

Reference:
    https://github.com/google-research/sound-separation/tree/master/models/bird_mixit

Requirements:
    uv pip install onnxruntime soundfile resampy numpy tqdm
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIXIT_SAMPLE_RATE = 22050  # Bird MixIT expects 22.05 kHz
MIXIT_BLOCK_SIZE = 22  # Learned basis window: 1ms at 22050 Hz (~22 samples)


def load_and_resample(audio_path: Path, target_sr: int = MIXIT_SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo to mono

    if sr != target_sr:
        import resampy

        audio = resampy.resample(audio, sr, target_sr)

    return audio


def separate(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    """Run MixIT separation on a mono waveform.

    Pads input to the nearest multiple of MIXIT_BLOCK_SIZE (the learned Conv1D
    kernel size) to avoid shape mismatches in the ONNX mixture consistency layer,
    then trims output back to original length.

    Args:
        session: ONNX inference session for Bird MixIT
        audio: (num_samples,) mono waveform at 22.05 kHz

    Returns:
        (num_sources, num_samples) separated channels at original length
    """
    original_len = len(audio)

    # Pad to nearest multiple of block size
    padded_len = int(np.ceil(original_len / MIXIT_BLOCK_SIZE) * MIXIT_BLOCK_SIZE)
    if padded_len != original_len:
        audio = np.pad(audio, (0, padded_len - original_len), mode="constant")

    # Model expects (1, 1, num_samples) — batch=1, mics=1, samples=N
    audio_3d = audio.reshape(1, 1, -1)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: audio_3d})[0]

    # Output is (1, num_sources, num_samples) — squeeze batch dim
    if output.ndim == 3:
        output = output.squeeze(0)

    # Trim to original length
    output = output[:, :original_len]

    return output  # (4, num_samples)


def get_labeled_filenames(data_dir: Path) -> set[str]:
    """Get filenames that appear in train_soundscapes_labels.csv."""
    labels_csv = data_dir / "train_soundscapes_labels.csv"
    if not labels_csv.exists():
        return set()

    filenames = set()
    with open(labels_csv) as f:
        next(f)  # skip header
        for line in f:
            fname = line.split(",")[0].strip()
            filenames.add(fname)
    return filenames


def main():
    parser = argparse.ArgumentParser(description="Batch MixIT separation of soundscapes")
    parser.add_argument(
        "--model",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx",
        help="Path to Bird MixIT ONNX model",
    )
    parser.add_argument(
        "--soundscape-dir",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes",
        help="Directory containing soundscape OGG files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026",
        help="Base data directory (for finding labels CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/separated",
        help="Output directory for separated channels (NPZ files)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Max files to process (0 = all)",
    )
    parser.add_argument(
        "--exclude-labeled",
        action="store_true",
        help="Exclude labeled soundscapes (process only unlabeled)",
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help="Process only labeled soundscapes (for validation)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of ONNX runtime threads (default: 4)",
    )
    args = parser.parse_args()

    soundscape_dir = Path(args.soundscape_dir)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get labeled filenames for filtering
    labeled_files = get_labeled_filenames(data_dir)
    logger.info(f"Found {len(labeled_files)} labeled soundscape files")

    # Collect files to process
    all_files = sorted(soundscape_dir.glob("*.ogg"))
    if args.exclude_labeled:
        files = [f for f in all_files if f.name not in labeled_files]
        logger.info(f"Excluding {len(labeled_files)} labeled files")
    elif args.only_labeled:
        files = [f for f in all_files if f.name in labeled_files]
        logger.info(f"Processing only {len(files)} labeled files")
    else:
        files = all_files

    if args.max_files > 0:
        files = files[: args.max_files]

    # Skip already-processed files
    todo = []
    for f in files:
        out_path = output_dir / f"{f.stem}.npz"
        if out_path.exists():
            continue
        todo.append(f)

    logger.info(
        f"Files to process: {len(todo)} ({len(files) - len(todo)} already done, "
        f"{len(files)} total selected)"
    )

    if not todo:
        logger.info("Nothing to do!")
        return

    # Load ONNX model
    logger.info(f"Loading ONNX model: {args.model}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.num_threads
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        args.model,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    logger.info(f"  Input: {input_info.name} {input_info.shape}")
    logger.info(f"  Output: {output_info.name} {output_info.shape}")

    # Process files
    times = []
    for audio_path in tqdm(todo, desc="Separating"):
        t0 = time.time()

        # Load and resample
        audio = load_and_resample(audio_path, MIXIT_SAMPLE_RATE)

        # Run separation
        channels = separate(session, audio)  # (4, num_samples)

        # Save as compressed NPZ
        out_path = output_dir / f"{audio_path.stem}.npz"
        np.savez_compressed(
            out_path,
            channels=channels,  # (4, num_samples) float32
            sample_rate=MIXIT_SAMPLE_RATE,
        )

        elapsed = time.time() - t0
        times.append(elapsed)

    # Summary
    if times:
        logger.info("\nSeparation complete!")
        logger.info(f"  Files processed: {len(times)}")
        logger.info(f"  Mean time per file: {np.mean(times):.2f}s")
        logger.info(f"  Total time: {sum(times) / 60:.1f} min")
        logger.info(f"  Output dir: {output_dir}")

        # Check output size
        total_size = sum(f.stat().st_size for f in output_dir.glob("*.npz"))
        logger.info(f"  Total output size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
