"""BirdSet data pipeline for SSAMBA training.

Self-contained data loading: HuggingFace datasets -> resampled audio -> on-the-fly
mel-spectrogram via Kaldi fbank. Outputs (B, 1, 128, target_frames) spectrograms
with padding masks for variable-length audio.

Training recordings are split into non-overlapping 30s chunks. Shorter/last chunks
are zero-padded, and padding masks track which tokens are real vs padded.
"""

import logging
import math
import os
import random
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

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


class MelSpectrogramTransform:
    """Kaldi fbank mel-spectrogram with AudioSet normalization.

    Follows the EAT / SSAMBA convention:
      - torchaudio.compliance.kaldi.fbank
      - Normalize: (fbank - mean) / (std * 2)
      - Pad or center-crop to target_frames

    Returns (spectrogram, real_frames) to support padding masks.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 3000,
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

    def __call__(self, waveform: torch.Tensor, real_samples: int = None):
        """Convert waveform to normalized mel-spectrogram.

        Args:
            waveform: (samples,) float tensor
            real_samples: Number of real audio samples (before padding).
                If None, all samples are considered real.

        Returns:
            Tuple of:
                (1, n_mels, target_frames) normalized mel-spectrogram
                int: number of real spectrogram frames
        """
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
        )  # (num_frames, n_mels)

        fbank = fbank.transpose(0, 1)  # (n_mels, num_frames)
        total_frames = fbank.shape[1]

        # Compute real frames from real_samples
        if real_samples is not None and real_samples < waveform.shape[0]:
            real_frames = min(
                (real_samples - 1) // self.frame_shift_samples + 1,
                total_frames,
            )
        else:
            real_frames = total_frames

        # Pad or crop to target_frames
        if fbank.shape[1] < self.target_frames:
            pad_length = self.target_frames - fbank.shape[1]
            fbank = torch.nn.functional.pad(fbank, (0, pad_length), value=0)
        elif fbank.shape[1] > self.target_frames:
            fbank = fbank[:, : self.target_frames]
            real_frames = min(real_frames, self.target_frames)

        # AudioSet normalization
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        return fbank.unsqueeze(0), real_frames  # (1, n_mels, target_frames), int


class StreamingBirdSetDataset(IterableDataset):
    """Streaming dataset that reads each audio file exactly once per epoch.

    Iterates over recordings, reads each file once, and yields all non-overlapping
    chunks. Workers are assigned disjoint subsets of recordings.

    For training (chunk=True): splits recordings into 30s windows.
    For validation (chunk=False): yields each clip as-is, padded to target_duration.
    """

    def __init__(
        self,
        hf_dataset,
        num_classes: int,
        target_sr: int = 32000,
        target_duration: float = 30.0,
        chunk: bool = False,
        shuffle: bool = False,
    ):
        self.dataset = hf_dataset
        self.num_classes = num_classes
        self.target_sr = target_sr
        self.target_samples = int(target_duration * target_sr)
        self.chunk = chunk
        self.shuffle = shuffle
        # Pre-count total chunks for logging / progress
        self._total_chunks = None

    def count_chunks(self):
        """Count total chunks across all recordings (for logging)."""
        if self._total_chunks is not None:
            return self._total_chunks
        if not self.chunk:
            self._total_chunks = len(self.dataset)
            return self._total_chunks
        total = 0
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            info = sf.info(sample["audio"]["path"])
            total_samples = int(info.frames * self.target_sr / info.samplerate)
            total += max(1, math.ceil(total_samples / self.target_samples))
        self._total_chunks = total
        return total

    def _make_labels(self, sample):
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        label_indices = sample.get("ebird_code_multilabel", [])
        if label_indices:
            for li in label_indices:
                if 0 <= li < self.num_classes:
                    labels[li] = 1.0
        return labels

    def _load_audio(self, sample):
        """Load and resample audio, returning (tensor, source_sr)."""
        audio_path = sample["audio"]["path"]
        audio, source_sr = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio_tensor = torch.from_numpy(audio).float()
        if source_sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=source_sr, new_freq=self.target_sr
            )
        return audio_tensor

    def _yield_chunks(self, audio_tensor, labels):
        """Yield non-overlapping chunks from a single recording."""
        total_samples = audio_tensor.shape[0]
        n_chunks = max(1, math.ceil(total_samples / self.target_samples))
        for c in range(n_chunks):
            start = c * self.target_samples
            chunk = audio_tensor[start : start + self.target_samples]
            real_samples = chunk.shape[0]
            if chunk.shape[0] < self.target_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.target_samples - chunk.shape[0])
                )
            yield {"audio": chunk, "labels": labels, "real_samples": real_samples}

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        # Split across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = math.ceil(len(indices) / worker_info.num_workers)
            start = worker_info.id * per_worker
            indices = indices[start : start + per_worker]

        for idx in indices:
            sample = self.dataset[idx]
            labels = self._make_labels(sample)
            audio_tensor = self._load_audio(sample)

            if self.chunk:
                yield from self._yield_chunks(audio_tensor, labels)
            else:
                real_samples = audio_tensor.shape[0]
                if audio_tensor.shape[0] < self.target_samples:
                    audio_tensor = torch.nn.functional.pad(
                        audio_tensor, (0, self.target_samples - audio_tensor.shape[0])
                    )
                yield {"audio": audio_tensor, "labels": labels, "real_samples": real_samples}


class BirdSetDataset(Dataset):
    """Map-style dataset wrapping a HuggingFace BirdSet split (no chunking).

    Used for validation where each clip is already 5s, loaded as-is and
    padded to target_duration. For chunked training, use StreamingBirdSetDataset.
    """

    def __init__(
        self,
        hf_dataset,
        num_classes: int,
        target_sr: int = 32000,
        target_duration: float = 30.0,
    ):
        self.dataset = hf_dataset
        self.num_classes = num_classes
        self.target_sr = target_sr
        self.target_samples = int(target_duration * target_sr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio_path = sample["audio"]["path"]
        audio, source_sr = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio_tensor = torch.from_numpy(audio).float()
        if source_sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=source_sr, new_freq=self.target_sr
            )

        real_samples = audio_tensor.shape[0]
        if audio_tensor.shape[0] < self.target_samples:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, self.target_samples - audio_tensor.shape[0])
            )

        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        label_indices = sample.get("ebird_code_multilabel", [])
        if label_indices:
            for li in label_indices:
                if 0 <= li < self.num_classes:
                    labels[li] = 1.0

        return {"audio": audio_tensor, "labels": labels, "real_samples": real_samples}


class CachedMelDataset(Dataset):
    """Dataset that loads pre-computed mel spectrograms from disk.

    Expects files named {idx:06d}.pt containing:
        mel: (1, n_mels, target_frames) float16 tensor
        labels: (num_classes,) float32 tensor
        real_frames: int
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.length = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        if self.length == 0:
            raise FileNotFoundError(f"No cached .pt files in {cache_dir}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.load(self.cache_dir / f"{idx:06d}.pt", weights_only=True)
        return {
            "mel": data["mel"].float(),
            "labels": data["labels"],
            "real_frames": int(data["real_frames"]),
        }


class BirdSetDataModule(pl.LightningDataModule):
    """Lightning DataModule for BirdSet with on-the-fly mel-spectrogram generation."""

    def __init__(
        self,
        dataset_name: str = "HSN",
        batch_size: int = 16,
        num_workers: int = 8,
        sample_rate: int = 32000,
        target_duration: float = 30.0,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 3000,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_frames = target_frames
        self.num_classes = BIRDSET_NUM_CLASSES[dataset_name]
        self.cache_dir = cache_dir

        self.mel_transform = MelSpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            frame_shift_ms=frame_shift_ms,
            target_frames=target_frames,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.cache_dir:
                train_cache = Path(self.cache_dir) / "train"
                val_cache = Path(self.cache_dir) / "val"
                self.train_dataset = CachedMelDataset(train_cache)
                self.val_dataset = CachedMelDataset(val_cache)
                logger.info(
                    f"Using cached mels: {len(self.train_dataset)} train, "
                    f"{len(self.val_dataset)} val"
                )
            else:
                logger.info(f"Loading {self.dataset_name} train split...")
                hf_train = load_dataset(
                    "DBD-research-group/BirdSet",
                    self.dataset_name,
                    split="train",
                    trust_remote_code=True,
                )
                self.train_dataset = StreamingBirdSetDataset(
                    hf_train,
                    num_classes=self.num_classes,
                    target_sr=self.sample_rate,
                    target_duration=self.target_duration,
                    chunk=True,
                    shuffle=True,
                )
                n_chunks = self.train_dataset.count_chunks()
                logger.info(f"Train: {n_chunks} chunks from {len(hf_train)} recordings (streaming)")

                logger.info(f"Loading {self.dataset_name} test_5s split for validation...")
                hf_val = load_dataset(
                    "DBD-research-group/BirdSet",
                    self.dataset_name,
                    split="test_5s",
                    trust_remote_code=True,
                )
                self.val_dataset = BirdSetDataset(
                    hf_val,
                    num_classes=self.num_classes,
                    target_sr=self.sample_rate,
                    target_duration=self.target_duration,
                )
                logger.info(f"Val: {len(self.val_dataset)} samples")

    def train_dataloader(self):
        is_cached = self.cache_dir is not None
        collate_fn = self._collate_cached if is_cached else self._collate_with_melspec
        # Cached = map-style (shuffle in DataLoader); streaming = shuffle internally
        shuffle = is_cached
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0 and is_cached,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        collate_fn = self._collate_cached if self.cache_dir else self._collate_with_melspec
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def _collate_cached(self, batch):
        """Collate pre-computed mel spectrograms.

        Returns:
            (mel_specs, labels, padding_mask) same format as _collate_with_melspec
        """
        mel_specs = torch.stack([item["mel"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        num_tokens = self.target_frames
        padding_mask = torch.zeros(len(batch), num_tokens, dtype=torch.bool)
        for i, item in enumerate(batch):
            padding_mask[i, : item["real_frames"]] = True

        return mel_specs, labels, padding_mask

    def _collate_with_melspec(self, batch):
        """Collate batch and apply mel-spectrogram transform on-the-fly.

        Returns:
            (mel_specs, labels, padding_mask) where:
                mel_specs: (B, 1, n_mels, target_frames)
                labels: (B, num_classes)
                padding_mask: (B, num_tokens) boolean, True for real tokens
        """
        audio = torch.stack([item["audio"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        real_samples_list = [item["real_samples"] for item in batch]

        mel_specs = []
        real_frames_list = []
        for i in range(audio.shape[0]):
            spec, real_frames = self.mel_transform(audio[i], real_samples_list[i])
            mel_specs.append(spec)
            real_frames_list.append(real_frames)

        mel_specs = torch.stack(mel_specs)

        # Build token-level padding mask
        # With patch_size=[128, 1], num_tokens = target_frames (1:1 mapping)
        num_tokens = self.mel_transform.target_frames
        padding_mask = torch.zeros(len(batch), num_tokens, dtype=torch.bool)
        for i, rf in enumerate(real_frames_list):
            padding_mask[i, :rf] = True

        return mel_specs, labels, padding_mask
