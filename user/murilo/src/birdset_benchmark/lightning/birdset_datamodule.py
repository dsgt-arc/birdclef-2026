"""PyTorch Lightning DataModule for BirdSet datasets.

Used by: Experiment 002 (Supervised ViT Baseline)

Integrates existing BirdSetDataset with Lightning's data loading patterns.
Generates mel-spectrograms on-the-fly in DataLoader workers.
"""

import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from birdset_benchmark.datasets import get_num_classes, load_birdset_dataset
from birdset_benchmark.preprocessing.mel_spectrogram import MelSpectrogramTransform
from birdset_benchmark.torch_data import BirdSetDataset, collate_fn

logger = logging.getLogger(__name__)


class BirdSetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for BirdSet with on-the-fly mel-spectrogram generation.

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
        self.train_dataset: BirdSetDataset | None = None
        self.val_dataset: BirdSetDataset | None = None
        self.test_dataset: BirdSetDataset | None = None

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
            self.train_dataset = BirdSetDataset(
                hf_train,
                num_classes=self.num_classes,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                return_labels=True,
                return_metadata=False,
            )
            logger.info(f"Train dataset: {len(self.train_dataset)} samples")

            # Load validation data (BirdSet uses 'test_5s' for validation)
            logger.info(f"Loading {self.dataset_name} test_5s split for validation...")
            hf_val = load_birdset_dataset(self.dataset_name, split="test_5s")
            self.val_dataset = BirdSetDataset(
                hf_val,
                num_classes=self.num_classes,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                return_labels=True,
                return_metadata=False,
            )
            logger.info(f"Val dataset: {len(self.val_dataset)} samples")

        if stage == "test" or stage == "predict":
            # Load test data (BirdSet uses 'test_5s' for standard evaluation)
            logger.info(f"Loading {self.dataset_name} test_5s split for testing...")
            hf_test = load_birdset_dataset(self.dataset_name, split="test_5s")
            self.test_dataset = BirdSetDataset(
                hf_test,
                num_classes=self.num_classes,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                return_labels=True,
                return_metadata=False,
            )
            logger.info(f"Test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with on-the-fly mel-spectrogram generation."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_with_melspec,
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
            collate_fn=self._collate_with_melspec,
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
            collate_fn=self._collate_with_melspec,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def _collate_with_melspec(self, batch):
        """Custom collate that applies vectorized mel-spectrogram transform.

        Args:
            batch: List of sample dicts from BirdSetDataset

        Returns:
            tuple: (mel_spectrograms, labels)
                - mel_spectrograms: Tensor of shape (batch, 1, n_mels, target_frames)
                - labels: Tensor of shape (batch, num_classes)
        """
        batched = collate_fn(batch)
        audio_batch = batched["audio"]  # (batch, samples)
        mel_specs = self.mel_transform(audio_batch)  # (batch, 1, n_mels, target_frames)
        labels = batched["labels"]  # (batch, num_classes)
        return mel_specs, labels
