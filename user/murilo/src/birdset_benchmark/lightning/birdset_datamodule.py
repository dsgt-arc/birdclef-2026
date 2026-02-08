"""PyTorch Lightning DataModule for BirdSet datasets.

Integrates existing BirdSetDataset with Lightning's data loading patterns.
Mel-spectrogram generation runs inside DataLoader workers for parallel I/O.
"""

import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from birdset_benchmark.datasets import get_num_classes, load_birdset_dataset
from birdset_benchmark.preprocessing.mel_spectrogram import MelSpectrogramTransform
from birdset_benchmark.torch_data import BirdSetDataset

logger = logging.getLogger(__name__)


class _MelDataset(Dataset):
    """Wraps BirdSetDataset to compute mel-spectrograms inside workers.

    Each DataLoader worker independently computes the mel-spectrogram for its
    samples, giving 8x parallelism over doing it in the main-process collate.
    """

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


class BirdSetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for BirdSet with on-the-fly mel-spectrogram generation.

    Mel-spectrograms are computed inside DataLoader workers (not in collate)
    so all 8 workers produce spectrograms in parallel.

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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

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
        self.train_dataset: _MelDataset | None = None
        self.val_dataset: _MelDataset | None = None
        self.test_dataset: _MelDataset | None = None

        # Number of classes (populated in setup())
        self.num_classes: int | None = None

    def setup(self, stage: str | None = None):
        """Load datasets and determine number of classes.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Get number of classes for this dataset
        self.num_classes = get_num_classes(self.dataset_name)
        logger.info(f"Loading {self.dataset_name} dataset with {self.num_classes} classes")

        if stage == "fit" or stage is None:
            # Load training data
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
            self.train_dataset = _MelDataset(base_train, self.mel_transform)
            logger.info(f"Train dataset: {len(self.train_dataset)} samples")

            # Load validation data (BirdSet uses 'test_5s' for validation)
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
            self.val_dataset = _MelDataset(base_val, self.mel_transform)
            logger.info(f"Val dataset: {len(self.val_dataset)} samples")

        if stage == "test" or stage == "predict":
            # Load test data (BirdSet uses 'test_5s' for standard evaluation)
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
            self.test_dataset = _MelDataset(base_test, self.mel_transform)
            logger.info(f"Test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with mel-spectrograms computed in workers."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,  # Drop incomplete batches for stable training
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
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
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
