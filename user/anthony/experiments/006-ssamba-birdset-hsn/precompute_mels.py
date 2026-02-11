"""Pre-compute mel spectrograms for faster training.

Reads each audio file exactly once, computes mel spectrograms for all chunks,
and saves as individual .pt files. Uses multiprocessing for parallelism.

Usage:
    uv run python precompute_mels.py --config config.yaml --cache-dir /path/to/cache
"""

import argparse
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.data import BIRDSET_NUM_CLASSES, MelSpectrogramTransform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _process_recording(args):
    """Process a single recording: read once, compute all chunk mels."""
    (
        audio_path,
        label_indices,
        num_classes,
        target_sr,
        target_samples,
        mel_kwargs,
        cache_dir,
        chunk_offset,
        do_chunk,
    ) = args

    # Read audio once
    audio, source_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=source_sr, new_freq=target_sr
        )

    # Labels
    labels = torch.zeros(num_classes, dtype=torch.float32)
    if label_indices:
        for li in label_indices:
            if 0 <= li < num_classes:
                labels[li] = 1.0

    mel_transform = MelSpectrogramTransform(**mel_kwargs)
    total_samples = audio_tensor.shape[0]

    if do_chunk:
        n_chunks = max(1, math.ceil(total_samples / target_samples))
    else:
        n_chunks = 1

    saved = 0
    for c in range(n_chunks):
        cache_path = cache_dir / f"{chunk_offset + c:06d}.pt"
        if cache_path.exists():
            continue

        if do_chunk:
            start = c * target_samples
            chunk = audio_tensor[start : start + target_samples]
            real_samples = chunk.shape[0]
            if chunk.shape[0] < target_samples:
                chunk = torch.nn.functional.pad(chunk, (0, target_samples - chunk.shape[0]))
        else:
            chunk = audio_tensor
            real_samples = total_samples
            if chunk.shape[0] < target_samples:
                chunk = torch.nn.functional.pad(chunk, (0, target_samples - chunk.shape[0]))

        mel, real_frames = mel_transform(chunk, real_samples)

        torch.save(
            {"mel": mel.half(), "labels": labels, "real_frames": real_frames},
            cache_path,
        )
        saved += 1

    return n_chunks, saved


def precompute_train(hf_dataset, num_classes, data_cfg, mel_kwargs, cache_dir, max_workers=8):
    """Pre-compute training chunks: read each file once, emit all chunks."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_sr = data_cfg["sample_rate"]
    target_samples = int(data_cfg["target_duration"] * target_sr)

    # Build work items: (recording_idx, chunk_offset)
    # First pass: count chunks per recording to assign global chunk indices
    logger.info("Counting chunks...")
    chunk_offsets = []
    offset = 0
    for i in range(len(hf_dataset)):
        chunk_offsets.append(offset)
        sample = hf_dataset[i]
        info = sf.info(sample["audio"]["path"])
        total = int(info.frames * target_sr / info.samplerate)
        n_chunks = max(1, math.ceil(total / target_samples))
        offset += n_chunks
    total_chunks = offset
    logger.info(f"Total: {total_chunks} chunks from {len(hf_dataset)} recordings")

    # Build args for parallel processing
    work_items = []
    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]
        work_items.append((
            sample["audio"]["path"],
            sample.get("ebird_code_multilabel", []),
            num_classes,
            target_sr,
            target_samples,
            mel_kwargs,
            cache_dir,
            chunk_offsets[i],
            True,  # do_chunk
        ))

    total_saved = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for n_chunks, saved in tqdm(
            executor.map(_process_recording, work_items),
            total=len(work_items),
            desc="Train recordings",
        ):
            total_saved += saved

    logger.info(f"Train: {total_chunks} chunks, {total_saved} newly saved")
    return total_chunks


def precompute_val(hf_dataset, num_classes, data_cfg, mel_kwargs, cache_dir, max_workers=8):
    """Pre-compute validation clips (no chunking, pad to target_duration)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_sr = data_cfg["sample_rate"]
    target_samples = int(data_cfg["target_duration"] * target_sr)

    work_items = []
    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]
        work_items.append((
            sample["audio"]["path"],
            sample.get("ebird_code_multilabel", []),
            num_classes,
            target_sr,
            target_samples,
            mel_kwargs,
            cache_dir,
            i,  # chunk_offset = index (1:1)
            False,  # do_chunk
        ))

    total_saved = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for n_chunks, saved in tqdm(
            executor.map(_process_recording, work_items),
            total=len(work_items),
            desc="Val clips",
        ):
            total_saved += saved

    logger.info(f"Val: {len(hf_dataset)} clips, {total_saved} newly saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset_name"]
    num_classes = BIRDSET_NUM_CLASSES[dataset_name]

    mel_kwargs = {
        "sample_rate": data_cfg["sample_rate"],
        "n_mels": data_cfg["n_mels"],
        "frame_shift_ms": data_cfg["frame_shift_ms"],
        "target_frames": data_cfg["target_frames"],
        "norm_mean": data_cfg["norm_mean"],
        "norm_std": data_cfg["norm_std"],
    }

    # Train split
    logger.info(f"Loading {dataset_name} train split...")
    hf_train = load_dataset(
        "DBD-research-group/BirdSet",
        dataset_name,
        split="train",
        trust_remote_code=True,
    )
    precompute_train(
        hf_train, num_classes, data_cfg, mel_kwargs,
        Path(args.cache_dir) / "train", max_workers=args.workers,
    )

    # Val split
    logger.info(f"Loading {dataset_name} test_5s split...")
    hf_val = load_dataset(
        "DBD-research-group/BirdSet",
        dataset_name,
        split="test_5s",
        trust_remote_code=True,
    )
    precompute_val(
        hf_val, num_classes, data_cfg, mel_kwargs,
        Path(args.cache_dir) / "val", max_workers=args.workers,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
