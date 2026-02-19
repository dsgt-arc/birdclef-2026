"""Streaming data pipeline for BirdSet clip-level classification.

Loads raw audio on-the-fly via partial reads (sf.read with start/frames),
splits recordings into non-overlapping 5s chunks, yielding ~6x more training
data than center-crop. No pre-loading into RAM — startup is near-instant.

Adapted from Anthony's exp 008 streaming pipeline, simplified for clip-level
classification (no SED/per-token labels).

Training: sequential non-overlapping 5s chunks from full recordings.
    Uses detected_events to zero-out labels for chunks with no bird activity.
Validation/Test: first 5s of each recording (matches existing behavior).
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, Sampler

from birdset_benchmark.datasets import get_num_classes, load_birdset_dataset

logger = logging.getLogger(__name__)


def _load_audio_chunk(audio_path, target_sr, start_sample=0, num_samples=None):
    """Load a chunk of audio and resample to target sample rate.

    Uses sf.read(start, frames) for partial reads to avoid loading entire files.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        start_sample: Start position in target sample rate
        num_samples: Number of samples to read in target sample rate (None = to end)
    """
    info = sf.info(audio_path)
    source_sr = info.samplerate

    # Log sample rates on first call to detect unnecessary resampling
    if not hasattr(_load_audio_chunk, "_logged"):
        logger.info(f"Audio source sample rate: {source_sr}, target: {target_sr}")
        _load_audio_chunk._logged = True

    # Convert target-domain positions to source-domain
    src_start = int(start_sample * source_sr / target_sr)
    if num_samples is not None:
        # Read a bit extra to avoid off-by-one after resampling
        src_frames = int(num_samples * source_sr / target_sr) + 2
    else:
        src_frames = -1  # read to end

    audio, _ = sf.read(audio_path, dtype="float32", start=src_start, frames=src_frames)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=source_sr, new_freq=target_sr
        )

    # Trim to exact requested length (resampling can produce ±1 sample)
    if num_samples is not None and audio_tensor.shape[0] > num_samples:
        audio_tensor = audio_tensor[:num_samples]

    return audio_tensor


def _parse_events(detected_events):
    """Parse detected_events into list of (start_time, end_time) tuples in seconds."""
    events = []
    if not detected_events:
        return events
    for event in detected_events:
        if isinstance(event, dict):
            start_t = event.get("start_time", event.get("start", 0))
            end_t = event.get("end_time", event.get("end", 0))
        elif isinstance(event, list | tuple) and len(event) >= 2:
            start_t, end_t = event[0], event[1]
        else:
            continue
        events.append((float(start_t), float(end_t)))
    return events


class StreamingBirdSetDataset(Dataset):
    """Streaming dataset: sequential 5s chunks from full recordings.

    Training: splits each recording into non-overlapping 5s chunks,
    yielding ~6x more training samples. Uses detected_events to filter
    out chunks with no bird vocalizations (cleaner labels).

    Validation: first 5s of each recording (matches existing behavior).
    """

    def __init__(
        self,
        hf_dataset,
        num_classes: int,
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        is_train: bool = True,
        manifest_cache_dir: str | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.is_train = is_train
        self.manifest_cache_dir = Path(manifest_cache_dir) if manifest_cache_dir else None

        # Build chunk index from audio manifest
        manifest = self._load_or_build_manifest()
        self.chunks = self._chunks_from_manifest(manifest)
        logger.info(
            f"{'Train' if is_train else 'Val'}: "
            f"{len(self.chunks)} chunks from {len(hf_dataset)} recordings"
        )

    def _manifest_path(self):
        """Cache path for audio manifest."""
        split = "train" if self.is_train else "val"
        return self.manifest_cache_dir / f"manifest_{split}_{len(self.hf_dataset)}.json"

    def _load_or_build_manifest(self):
        """Load or build audio manifest: list of (rec_idx, path, native_frames, native_sr).

        The manifest captures the slow I/O (HF deserialization + sf.info) and is
        reusable across any sample_rate or chunk_duration.
        """
        if self.manifest_cache_dir:
            path = self._manifest_path()
            if path.exists():
                logger.info(f"Loading audio manifest from cache: {path}")
                with open(path) as f:
                    return json.load(f)

        logger.info(f"Building audio manifest for {len(self.hf_dataset)} recordings...")
        manifest = []
        for rec_idx in range(len(self.hf_dataset)):
            sample = self.hf_dataset[rec_idx]
            audio_path = sample["audio"]["path"]
            info = sf.info(audio_path)
            manifest.append((rec_idx, audio_path, info.frames, info.samplerate))

        if self.manifest_cache_dir:
            path = self._manifest_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(manifest, f)
            logger.info(f"Saved audio manifest to cache: {path}")

        return manifest

    def _chunks_from_manifest(self, manifest):
        """Compute chunk index from manifest — pure arithmetic, instant."""
        chunks = []
        for rec_idx, _path, native_frames, native_sr in manifest:
            total_samples = int(native_frames * self.sample_rate / native_sr)

            if self.is_train:
                pos = 0
                while pos < total_samples:
                    end = min(pos + self.chunk_samples, total_samples)
                    real_samples = end - pos
                    chunks.append((rec_idx, pos, end, real_samples))
                    pos += self.chunk_samples
            else:
                real_samples = min(total_samples, self.chunk_samples)
                chunks.append((rec_idx, 0, real_samples, real_samples))

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        rec_idx, chunk_start, chunk_end, real_samples = self.chunks[idx]
        sample = self.hf_dataset[rec_idx]

        # Load only the chunk we need (partial read, no full-file loading)
        audio_path = sample["audio"]["path"]
        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=chunk_start,
            num_samples=real_samples,
        )
        # Pad to chunk_samples if needed
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        # Multi-hot labels from ebird_code_multilabel
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        ebird_labels = sample.get("ebird_code_multilabel")
        if ebird_labels is not None:
            for label_idx in ebird_labels:
                li = int(label_idx)
                if 0 <= li < self.num_classes:
                    labels[li] = 1.0

        # Event-aware filtering (train only):
        # If detected_events exists AND no event overlaps this chunk, zero out labels
        if self.is_train:
            detected_events = sample.get("detected_events")
            events = _parse_events(detected_events)
            if events:
                chunk_start_s = chunk_start / self.sample_rate
                chunk_end_s = chunk_end / self.sample_rate
                has_overlap = any(es < chunk_end_s and ee > chunk_start_s for es, ee in events)
                if not has_overlap:
                    labels = torch.zeros_like(labels)

        return waveform, labels


class RecordingShuffleSampler(Sampler):
    """Shuffle recording order but yield chunks within each recording sequentially.

    This gives each DataLoader worker sequential file access (good for OS page cache
    and network storage), while still shuffling across recordings each epoch.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        # Group chunk indices by recording_idx
        rec_to_indices = defaultdict(list)
        for i, (rec_idx, *_) in enumerate(chunks):
            rec_to_indices[rec_idx].append(i)
        self.rec_groups = list(rec_to_indices.values())

    def __iter__(self):
        # Shuffle recording order each epoch, keep chunks sequential within
        order = torch.randperm(len(self.rec_groups)).tolist()
        for i in order:
            yield from self.rec_groups[i]

    def __len__(self):
        return len(self.chunks)


class StreamingBirdSetDataModule(pl.LightningDataModule):
    """Lightning DataModule for streaming BirdSet audio — clip-level classification.

    Replaces BirdSetDataModule's center-crop + pre-load approach with on-the-fly
    partial reads and full-recording chunking. Near-instant startup, ~6x more
    training data per epoch.

    Returns (waveform, labels) tuples — compatible with SupervisedViTModule.
    """

    def __init__(
        self,
        dataset_name: str = "HSN",
        batch_size: int = 128,
        num_workers: int = 8,
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 8,
        manifest_cache_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.num_classes = get_num_classes(dataset_name)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.manifest_cache_dir = manifest_cache_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            logger.info(f"Loading {self.dataset_name} for streaming...")
            hf_train = load_birdset_dataset(self.dataset_name, split="train")
            self.train_dataset = StreamingBirdSetDataset(
                hf_dataset=hf_train,
                num_classes=self.num_classes,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                is_train=True,
                manifest_cache_dir=self.manifest_cache_dir,
            )

            hf_val = load_birdset_dataset(self.dataset_name, split="test_5s")
            self.val_dataset = StreamingBirdSetDataset(
                hf_dataset=hf_val,
                num_classes=self.num_classes,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                is_train=False,
                manifest_cache_dir=self.manifest_cache_dir,
            )

        if stage == "test" or stage == "predict":
            hf_test = load_birdset_dataset(self.dataset_name, split="test_5s")
            self.test_dataset = StreamingBirdSetDataset(
                hf_dataset=hf_test,
                num_classes=self.num_classes,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                is_train=False,
                manifest_cache_dir=self.manifest_cache_dir,
            )

    def train_dataloader(self):
        sampler = RecordingShuffleSampler(self.train_dataset.chunks)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
