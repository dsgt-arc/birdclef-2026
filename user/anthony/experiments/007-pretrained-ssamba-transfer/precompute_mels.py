"""Pre-compute event-centered 10s mel spectrograms for experiment 007.

Extracts event-centered windows from BirdSet HSN train recordings using
detected_events, plus negative (non-event) windows. Validation clips (5s)
are padded to 10s.

Usage:
    uv run python precompute_mels.py --config config-default-16x16.yaml \
        --cache-dir /path/to/cache/007-hsn-events
"""

import argparse
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.data import BIRDSET_NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class MelSpectrogramTransform:
    """Kaldi fbank mel-spectrogram with AudioSet normalization."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 1000,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_shift = frame_shift_ms
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.frame_shift_samples = int(sample_rate * frame_shift_ms / 1000)

    def __call__(self, waveform, real_samples=None):
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.n_mels,
            dither=0.0,
            frame_shift=self.frame_shift,
        )
        fbank = fbank.transpose(0, 1)  # (n_mels, num_frames)
        total_frames = fbank.shape[1]

        if real_samples is not None and real_samples < waveform.shape[0]:
            real_frames = min(
                (real_samples - 1) // self.frame_shift_samples + 1,
                total_frames,
            )
        else:
            real_frames = total_frames

        if fbank.shape[1] < self.target_frames:
            pad_length = self.target_frames - fbank.shape[1]
            fbank = torch.nn.functional.pad(fbank, (0, pad_length), value=0)
        elif fbank.shape[1] > self.target_frames:
            fbank = fbank[:, : self.target_frames]
            real_frames = min(real_frames, self.target_frames)

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        return fbank.unsqueeze(0), real_frames


def _load_audio(audio_path, target_sr):
    """Load and resample audio."""
    audio, source_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=source_sr, new_freq=target_sr
        )
    return audio_tensor


def _extract_window(audio_tensor, center_sample, window_samples):
    """Extract a window centered on center_sample, clamped to bounds."""
    total = audio_tensor.shape[0]
    start = center_sample - window_samples // 2
    end = start + window_samples

    # Clamp to recording bounds
    if start < 0:
        start = 0
        end = window_samples
    if end > total:
        end = total
        start = max(0, end - window_samples)

    chunk = audio_tensor[start:end]
    real_samples = chunk.shape[0]

    # Pad if still shorter than window
    if chunk.shape[0] < window_samples:
        chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[0]))

    return chunk, real_samples


def _process_train_recording(args):
    """Process a single training recording: extract event-centered + negative windows."""
    (
        audio_path,
        label_indices,
        detected_events,
        num_classes,
        target_sr,
        window_samples,
        mel_kwargs,
        cache_dir,
        sample_offset,
    ) = args

    audio_tensor = _load_audio(audio_path, target_sr)
    total_samples = audio_tensor.shape[0]
    duration_s = total_samples / target_sr

    labels = torch.zeros(num_classes, dtype=torch.float32)
    if label_indices:
        for li in label_indices:
            if 0 <= li < num_classes:
                labels[li] = 1.0

    mel_transform = MelSpectrogramTransform(**mel_kwargs)
    saved = 0
    idx = sample_offset

    # Parse detected_events and extract event-centered windows
    event_midpoints = []
    if detected_events:
        for event in detected_events:
            # detected_events is a list of dicts with start_time and end_time (seconds)
            # or it could be structured differently — handle both formats
            if isinstance(event, dict):
                start_t = event.get("start_time", event.get("start", 0))
                end_t = event.get("end_time", event.get("end", 0))
            elif isinstance(event, (list, tuple)) and len(event) >= 2:
                start_t, end_t = event[0], event[1]
            else:
                continue

            mid_t = (start_t + end_t) / 2.0
            mid_sample = int(mid_t * target_sr)
            event_midpoints.append(mid_sample)

    if event_midpoints:
        # Deduplicate events that are very close (within 5s of each other)
        event_midpoints.sort()
        deduped = [event_midpoints[0]]
        for mp in event_midpoints[1:]:
            if (mp - deduped[-1]) / target_sr > 5.0:
                deduped.append(mp)
        event_midpoints = deduped

        # Extract event-centered windows
        for mid_sample in event_midpoints:
            cache_path = cache_dir / f"{idx:06d}.pt"
            if not cache_path.exists():
                chunk, real_samples = _extract_window(audio_tensor, mid_sample, window_samples)
                mel, real_frames = mel_transform(chunk, real_samples)
                torch.save(
                    {"mel": mel.half(), "labels": labels, "real_frames": real_frames},
                    cache_path,
                )
            saved += 1
            idx += 1

        # Sample negative windows (non-event regions)
        # Find time regions at least 10s away from any event
        window_s = window_samples / target_sr
        event_times_s = [mp / target_sr for mp in event_midpoints]

        # Generate candidate negative centers (every 5s, check distance from events)
        neg_candidates = []
        for t in np.arange(window_s / 2, duration_s - window_s / 2, 5.0):
            min_dist = min(abs(t - et) for et in event_times_s)
            if min_dist >= window_s:  # At least 10s from any event
                neg_candidates.append(int(t * target_sr))

        # Sample up to same number of negatives as events
        n_neg = min(len(neg_candidates), len(event_midpoints))
        if n_neg > 0 and neg_candidates:
            rng = np.random.RandomState(sample_offset)  # deterministic per recording
            chosen = rng.choice(len(neg_candidates), size=n_neg, replace=False)
            neg_labels = torch.zeros(num_classes, dtype=torch.float32)

            for ci in chosen:
                cache_path = cache_dir / f"{idx:06d}.pt"
                if not cache_path.exists():
                    chunk, real_samples = _extract_window(
                        audio_tensor, neg_candidates[ci], window_samples
                    )
                    mel, real_frames = mel_transform(chunk, real_samples)
                    torch.save(
                        {"mel": mel.half(), "labels": neg_labels, "real_frames": real_frames},
                        cache_path,
                    )
                saved += 1
                idx += 1
    else:
        # No detected_events: fall back to single centered window with recording labels
        cache_path = cache_dir / f"{idx:06d}.pt"
        if not cache_path.exists():
            mid_sample = total_samples // 2
            chunk, real_samples = _extract_window(audio_tensor, mid_sample, window_samples)
            mel, real_frames = mel_transform(chunk, real_samples)
            torch.save(
                {"mel": mel.half(), "labels": labels, "real_frames": real_frames},
                cache_path,
            )
        saved += 1
        idx += 1

    total_windows = idx - sample_offset
    return total_windows, saved


def _process_val_clip(args):
    """Process a single validation clip: pad 5s to 10s."""
    (
        audio_path,
        label_indices,
        num_classes,
        target_sr,
        window_samples,
        mel_kwargs,
        cache_dir,
        idx,
    ) = args

    audio_tensor = _load_audio(audio_path, target_sr)
    real_samples = audio_tensor.shape[0]

    if audio_tensor.shape[0] < window_samples:
        audio_tensor = torch.nn.functional.pad(
            audio_tensor, (0, window_samples - audio_tensor.shape[0])
        )

    labels = torch.zeros(num_classes, dtype=torch.float32)
    if label_indices:
        for li in label_indices:
            if 0 <= li < num_classes:
                labels[li] = 1.0

    mel_transform = MelSpectrogramTransform(**mel_kwargs)
    cache_path = cache_dir / f"{idx:06d}.pt"

    if not cache_path.exists():
        mel, real_frames = mel_transform(audio_tensor, real_samples)
        torch.save(
            {"mel": mel.half(), "labels": labels, "real_frames": real_frames},
            cache_path,
        )
        return 1, 1
    return 1, 0


def _count_windows_per_recording(sample, target_sr, window_samples):
    """Count how many windows (event + negative) a recording will produce."""
    audio_path = sample["audio"]["path"]
    info = sf.info(audio_path)
    duration_s = info.frames / info.samplerate
    total_samples = int(info.frames * target_sr / info.samplerate)

    detected_events = sample.get("detected_events")
    window_s = window_samples / target_sr

    if not detected_events:
        return 1  # fallback single window

    # Count events after deduplication
    event_times = []
    for event in detected_events:
        if isinstance(event, dict):
            start_t = event.get("start_time", event.get("start", 0))
            end_t = event.get("end_time", event.get("end", 0))
        elif isinstance(event, (list, tuple)) and len(event) >= 2:
            start_t, end_t = event[0], event[1]
        else:
            continue
        event_times.append((start_t + end_t) / 2.0)

    if not event_times:
        return 1

    event_times.sort()
    deduped = [event_times[0]]
    for t in event_times[1:]:
        if t - deduped[-1] > 5.0:
            deduped.append(t)

    n_events = len(deduped)

    # Count negatives
    neg_count = 0
    for t in np.arange(window_s / 2, duration_s - window_s / 2, 5.0):
        min_dist = min(abs(t - et) for et in deduped)
        if min_dist >= window_s:
            neg_count += 1
    n_neg = min(neg_count, n_events)

    return n_events + n_neg


def precompute_train(hf_dataset, num_classes, data_cfg, mel_kwargs, cache_dir, max_workers=8):
    """Pre-compute event-centered training windows."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_sr = data_cfg["sample_rate"]
    window_samples = int(data_cfg["target_duration"] * target_sr)

    # Count windows per recording for offset assignment
    logger.info("Counting windows per recording...")
    offsets = []
    offset = 0
    for i in tqdm(range(len(hf_dataset)), desc="Counting"):
        offsets.append(offset)
        n = _count_windows_per_recording(hf_dataset[i], target_sr, window_samples)
        offset += n
    total_expected = offset
    logger.info(f"Expected ~{total_expected} windows from {len(hf_dataset)} recordings")

    # Check if detected_events exists
    sample0 = hf_dataset[0]
    has_events = "detected_events" in sample0 and sample0["detected_events"] is not None
    if has_events:
        logger.info(f"detected_events field found, sample: {str(sample0['detected_events'])[:200]}")
    else:
        logger.warning("detected_events NOT found in dataset — falling back to centered windows")

    work_items = []
    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]
        work_items.append((
            sample["audio"]["path"],
            sample.get("ebird_code_multilabel", []),
            sample.get("detected_events"),
            num_classes,
            target_sr,
            window_samples,
            mel_kwargs,
            cache_dir,
            offsets[i],
        ))

    total_saved = 0
    total_windows = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for n_windows, saved in tqdm(
            executor.map(_process_train_recording, work_items),
            total=len(work_items),
            desc="Train recordings",
        ):
            total_windows += n_windows
            total_saved += saved

    logger.info(f"Train: {total_windows} windows, {total_saved} newly saved")
    return total_windows


def precompute_val(hf_dataset, num_classes, data_cfg, mel_kwargs, cache_dir, max_workers=8):
    """Pre-compute validation clips (5s padded to 10s)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_sr = data_cfg["sample_rate"]
    window_samples = int(data_cfg["target_duration"] * target_sr)

    work_items = []
    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]
        work_items.append((
            sample["audio"]["path"],
            sample.get("ebird_code_multilabel", []),
            num_classes,
            target_sr,
            window_samples,
            mel_kwargs,
            cache_dir,
            i,
        ))

    total_saved = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for n_clips, saved in tqdm(
            executor.map(_process_val_clip, work_items),
            total=len(work_items),
            desc="Val clips",
        ):
            total_saved += saved

    logger.info(f"Val: {len(hf_dataset)} clips, {total_saved} newly saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config-default-16x16.yaml")
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
