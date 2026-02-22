"""PyTorch Lightning DataModule for BirdSet datasets.

Supports three data-loading modes controlled by ``cache_mel_specs`` and
``waveform_only``:

* **waveform_only=True** — pre-loads all waveforms into RAM during
  ``setup()`` and returns them for GPU-side mel computation via nnAudio.
  Zero disk I/O during training. Fastest option.
* **cache_mel_specs=True** — pre-computes mel spectrograms on CPU during
  ``setup()`` and caches in RAM. No disk I/O during training.
* **Both False (default)** — mel spectrograms computed on-the-fly in
  DataLoader workers on CPU.
"""

import logging
import time

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from birdclef.datasets import get_num_classes, load_birdset_dataset
from birdclef.preprocessing.mel_spectrogram import MelSpectrogramTransform
from birdclef.torch_data import BirdSetDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------


class _MelDataset(Dataset):
    """Computes mel spectrograms on-the-fly inside DataLoader workers."""

    def __init__(self, base_dataset: BirdSetDataset, mel_transform: MelSpectrogramTransform):
        self.base_dataset = base_dataset
        self.mel_transform = mel_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.base_dataset[idx]
        mel = self.mel_transform(sample["audio"])  # (1, n_mels, target_frames)
        labels = sample["labels"]  # (num_classes,)
        return mel, labels


class _CachedMelDataset(Dataset):
    """Pre-computed mel spectrograms served from memory. Zero disk I/O."""

    def __init__(self, mel_specs: torch.Tensor, labels: torch.Tensor):
        self.mel_specs = mel_specs  # (N, 1, n_mels, target_frames)
        self.labels = labels  # (N, num_classes)

    def __len__(self) -> int:
        return self.mel_specs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mel_specs[idx], self.labels[idx]


class _CachedWaveformDataset(Dataset):
    """Pre-loaded waveforms served from memory for GPU-side mel (nnAudio).

    All audio is read from disk once during construction and cached as
    contiguous tensors. Zero disk I/O during training.
    """

    def __init__(self, waveforms: torch.Tensor, labels: torch.Tensor):
        self.waveforms = waveforms  # (N, num_samples)
        self.labels = labels  # (N, num_classes)

    def __len__(self) -> int:
        return self.waveforms.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.waveforms[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class BirdSetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for BirdSet datasets.

    Args:
        dataset_name: BirdSet dataset name (e.g., "HSN", "XCM")
        batch_size: Batch size for training/validation/test
        num_workers: Number of DataLoader workers
        target_sr: Target sample rate for audio (default: 32000 Hz)
        target_duration: Target duration in seconds (default: 5.0)
        n_mels: Number of mel frequency bins (default: 128)
        frame_shift_ms: Frame shift in milliseconds (default: 10)
        target_frames: Target number of time frames (default: 500 for ~5s)
        norm_mean: Normalization mean (default: AudioSet mean)
        norm_std: Normalization std (default: AudioSet std)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        persistent_workers: Keep workers alive between epochs (default: True)
        prefetch_factor: Number of batches to prefetch per worker (default: 4)
        cache_mel_specs: Pre-compute all mel spectrograms and cache in RAM.
            Eliminates disk I/O during training but requires the full dataset
            to fit in memory. (default: False)
        waveform_only: Return raw waveforms instead of mel spectrograms.
            Use this when the model has a GPU mel front-end (e.g., nnAudio).
            Takes precedence over cache_mel_specs. (default: False)
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 8,
        target_sr: int = 32000,
        target_duration: float = 5.0,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 500,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        cache_mel_specs: bool = False,
        waveform_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.cache_mel_specs = cache_mel_specs
        self.waveform_only = waveform_only

        # Audio preprocessing params
        self.target_sr = target_sr
        self.target_duration = target_duration

        # Mel-spectrogram transform
        self.mel_transform = MelSpectrogramTransform(
            sample_rate=target_sr,
            n_mels=n_mels,
            frame_shift_ms=frame_shift_ms,
            target_frames=target_frames,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # Datasets (populated in setup())
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # Number of classes (populated in setup())
        self.num_classes: int | None = None

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _precompute_waveforms(
        self, base_dataset: BirdSetDataset, split_name: str
    ) -> _CachedWaveformDataset:
        """Pre-load all waveforms into RAM. Zero disk I/O during training."""
        n = len(base_dataset)
        logger.info(f"Pre-loading waveforms for {split_name} ({n} samples)...")
        t0 = time.time()

        wav_list = []
        label_list = []
        for i in range(n):
            sample = base_dataset[i]
            wav_list.append(sample["audio"])
            label_list.append(sample["labels"])

            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                logger.info(f"  {split_name}: {i + 1}/{n} samples ({elapsed:.1f}s)")

        wav_tensor = torch.stack(wav_list)  # (N, num_samples)
        label_tensor = torch.stack(label_list)  # (N, num_classes)

        elapsed = time.time() - t0
        mem_gb = wav_tensor.nbytes / (1024**3)
        logger.info(f"  {split_name}: done — {n} samples, {mem_gb:.2f} GB, {elapsed:.1f}s")

        return _CachedWaveformDataset(wav_tensor, label_tensor)

    def _precompute_mels(self, base_dataset: BirdSetDataset, split_name: str) -> _CachedMelDataset:
        """Pre-compute all mel spectrograms and cache in memory."""
        n = len(base_dataset)
        logger.info(f"Pre-computing mel spectrograms for {split_name} ({n} samples)...")
        t0 = time.time()

        mel_list = []
        label_list = []
        for i in range(n):
            sample = base_dataset[i]
            mel = self.mel_transform(sample["audio"])  # (1, n_mels, target_frames)
            mel_list.append(mel)
            label_list.append(sample["labels"])

            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                logger.info(f"  {split_name}: {i + 1}/{n} samples ({elapsed:.1f}s)")

        mel_tensor = torch.stack(mel_list)  # (N, 1, n_mels, target_frames)
        label_tensor = torch.stack(label_list)  # (N, num_classes)

        elapsed = time.time() - t0
        mem_gb = mel_tensor.nbytes / (1024**3)
        logger.info(f"  {split_name}: done — {n} samples, {mem_gb:.2f} GB, {elapsed:.1f}s")

        return _CachedMelDataset(mel_tensor, label_tensor)

    def _wrap_dataset(self, base_dataset: BirdSetDataset, split_name: str) -> Dataset:
        """Wrap a base dataset — waveform, cached, or on-the-fly depending on config."""
        if self.waveform_only:
            return self._precompute_waveforms(base_dataset, split_name)
        if self.cache_mel_specs:
            return self._precompute_mels(base_dataset, split_name)
        return _MelDataset(base_dataset, self.mel_transform)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None):
        """Load datasets and optionally pre-compute mel spectrograms.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        self.num_classes = get_num_classes(self.dataset_name)
        if self.waveform_only:
            mode = "cached waveform (GPU mel via nnAudio)"
        elif self.cache_mel_specs:
            mode = "cached"
        else:
            mode = "on-the-fly"
        logger.info(
            f"Loading {self.dataset_name} dataset with {self.num_classes} classes "
            f"(mode: {mode})"
        )

        if stage == "fit" or stage is None:
            logger.info(f"Loading {self.dataset_name} train split...")
            hf_train = load_birdset_dataset(self.dataset_name, split="train")
            base_train = BirdSetDataset(
                hf_train,
                num_classes=self.num_classes,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                return_labels=True,
                return_metadata=False,
            )
            self.train_dataset = self._wrap_dataset(base_train, "train")
            logger.info(f"Train dataset: {len(self.train_dataset)} samples")

            logger.info(f"Loading {self.dataset_name} test_5s split for validation...")
            hf_val = load_birdset_dataset(self.dataset_name, split="test_5s")
            base_val = BirdSetDataset(
                hf_val,
                num_classes=self.num_classes,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                return_labels=True,
                return_metadata=False,
            )
            self.val_dataset = self._wrap_dataset(base_val, "val")
            logger.info(f"Val dataset: {len(self.val_dataset)} samples")

        if stage == "test" or stage == "predict":
            logger.info(f"Loading {self.dataset_name} test_5s split for testing...")
            hf_test = load_birdset_dataset(self.dataset_name, split="test_5s")
            base_test = BirdSetDataset(
                hf_test,
                num_classes=self.num_classes,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                return_labels=True,
                return_metadata=False,
            )
            self.test_dataset = self._wrap_dataset(base_test, "test")
            logger.info(f"Test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
