"""BirdSet streaming data pipeline with per-token SED labels.

Loads raw audio on-the-fly, splits into sequential 10s chunks, and constructs
per-token labels from detected_events for SED training.

Training: sequential non-overlapping 10s chunks from full recordings
Validation: 5s clips padded to 10s (clip-level labels only)
"""

import logging

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Number of species classes per BirdSet dataset
BIRDSET_NUM_CLASSES = {
    "HSN": 21,
    "PER": 132,
    "NES": 43,
    "UHH": 27,
    "POW": 48,
    "SSW": 81,
    "SNE": 52,
    "NBP": 51,
}


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
        elif isinstance(event, (list, tuple)) and len(event) >= 2:
            start_t, end_t = event[0], event[1]
        else:
            continue
        events.append((float(start_t), float(end_t)))
    return events


def _build_token_labels(events, chunk_start_s, num_tokens, hop_s, num_classes, label_vec):
    """Build per-token multi-label targets from detected events.

    A token at position i covers time [chunk_start + i*hop, chunk_start + (i+1)*hop].
    If any event overlaps this interval, the token gets the recording's label vector.

    Args:
        events: List of (start_time, end_time) in seconds
        chunk_start_s: Start time of this chunk in seconds
        num_tokens: Number of tokens (T)
        hop_s: Time duration per token in seconds
        num_classes: C
        label_vec: (C,) recording-level label vector

    Returns:
        token_labels: (T, C) tensor
    """
    token_labels = torch.zeros(num_tokens, num_classes, dtype=torch.float32)
    if not events:
        return token_labels

    for i in range(num_tokens):
        tok_start = chunk_start_s + i * hop_s
        tok_end = tok_start + hop_s
        for ev_start, ev_end in events:
            if ev_start < tok_end and ev_end > tok_start:
                token_labels[i] = label_vec
                break

    return token_labels


class BirdSetStreamingDataset(Dataset):
    """Dataset that loads raw audio on-the-fly and returns waveform chunks with SED labels.

    Training: sequential non-overlapping 10s chunks from full recordings.
    Validation: 5s clips padded to 10s, clip-level labels only (no token labels).
    """

    def __init__(
        self,
        hf_dataset,
        num_classes: int,
        sample_rate: int = 32000,
        chunk_duration: float = 10.0,
        num_tokens: int = 1000,
        is_train: bool = True,
    ):
        self.hf_dataset = hf_dataset
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.num_tokens = num_tokens
        self.hop_s = chunk_duration / num_tokens  # seconds per token
        self.is_train = is_train

        # Build chunk index: list of (recording_idx, chunk_start_sample, chunk_end_sample, real_samples)
        self.chunks = self._build_chunk_index()
        logger.info(f"{'Train' if is_train else 'Val'} dataset: {len(self.chunks)} chunks from {len(hf_dataset)} recordings")

    def _build_chunk_index(self):
        """Pre-index all chunks from all recordings."""
        chunks = []
        for rec_idx in range(len(self.hf_dataset)):
            sample = self.hf_dataset[rec_idx]
            audio_info = sample["audio"]

            if self.is_train:
                # Get duration from audio metadata
                path = audio_info["path"]
                info = sf.info(path)
                total_samples = int(info.frames * self.sample_rate / info.samplerate)

                # Sequential non-overlapping chunks
                pos = 0
                while pos < total_samples:
                    end = min(pos + self.chunk_samples, total_samples)
                    real_samples = end - pos
                    chunks.append((rec_idx, pos, end, real_samples))
                    pos += self.chunk_samples
            else:
                # Validation: single chunk per clip (5s padded to 10s)
                path = audio_info["path"]
                info = sf.info(path)
                total_samples = int(info.frames * self.sample_rate / info.samplerate)
                real_samples = min(total_samples, self.chunk_samples)
                chunks.append((rec_idx, 0, min(total_samples, self.chunk_samples), real_samples))

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        rec_idx, chunk_start, chunk_end, real_samples = self.chunks[idx]
        sample = self.hf_dataset[rec_idx]

        # Load only the chunk we need (partial read, no full-file loading)
        audio_path = sample["audio"]["path"]
        waveform = _load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=chunk_start, num_samples=real_samples,
        )
        # Pad to chunk_samples if needed
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        # Build clip-level labels
        clip_labels = torch.zeros(self.num_classes, dtype=torch.float32)
        # BirdSet ebird_code_multilabel is a list of class indices, e.g. [2, 4, 6]
        ebird_labels = sample.get("ebird_code_multilabel")
        if ebird_labels is not None:
            for idx in ebird_labels:
                li = int(idx)
                if 0 <= li < self.num_classes:
                    clip_labels[li] = 1.0

        # Build per-token labels (train only)
        if self.is_train:
            detected_events = sample.get("detected_events")
            events = _parse_events(detected_events)
            chunk_start_s = chunk_start / self.sample_rate
            token_labels = _build_token_labels(
                events, chunk_start_s, self.num_tokens,
                self.hop_s, self.num_classes, clip_labels,
            )
            # If any token is active, clip_labels stays as recording labels.
            # If no token is active (chunk is entirely outside events), zero out clip labels.
            if token_labels.sum() == 0:
                clip_labels = torch.zeros_like(clip_labels)
        else:
            token_labels = torch.zeros(self.num_tokens, self.num_classes, dtype=torch.float32)

        # Compute real tokens from real samples
        # Each token covers hop_s seconds = hop_s * sample_rate samples
        hop_samples = self.chunk_samples // self.num_tokens
        real_tokens = min(
            (real_samples + hop_samples - 1) // hop_samples,
            self.num_tokens,
        )

        return {
            "waveform": waveform,
            "clip_labels": clip_labels,
            "token_labels": token_labels,
            "real_tokens": int(real_tokens),
        }


def _collate_fn(batch):
    """Collate batch with padding masks.

    Returns:
        waveforms: (B, num_samples)
        clip_labels: (B, C)
        token_labels: (B, T, C)
        padding_mask: (B, T) boolean, True for real tokens
    """
    waveforms = torch.stack([item["waveform"] for item in batch])
    clip_labels = torch.stack([item["clip_labels"] for item in batch])
    token_labels = torch.stack([item["token_labels"] for item in batch])

    # Build padding mask
    num_tokens = token_labels.shape[1]
    masks = []
    for item in batch:
        real_tokens = item["real_tokens"]
        mask = torch.arange(num_tokens) < real_tokens
        masks.append(mask)
    padding_mask = torch.stack(masks)

    return waveforms, clip_labels, token_labels, padding_mask


class BirdSetStreamingDataModule(pl.LightningDataModule):
    """Lightning DataModule for streaming BirdSet audio with SED labels."""

    def __init__(
        self,
        dataset_name: str = "HSN",
        batch_size: int = 16,
        num_workers: int = 8,
        sample_rate: int = 32000,
        chunk_duration: float = 10.0,
        num_tokens: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.num_tokens = num_tokens
        self.num_classes = BIRDSET_NUM_CLASSES[dataset_name]

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            ds = load_dataset(
                f"DBD-research-group/BirdSet",
                self.dataset_name,
                trust_remote_code=True,
            )
            self.train_dataset = BirdSetStreamingDataset(
                hf_dataset=ds["train"],
                num_classes=self.num_classes,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                num_tokens=self.num_tokens,
                is_train=True,
            )
            self.val_dataset = BirdSetStreamingDataset(
                hf_dataset=ds["test_5s"],
                num_classes=self.num_classes,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                num_tokens=self.num_tokens,
                is_train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
