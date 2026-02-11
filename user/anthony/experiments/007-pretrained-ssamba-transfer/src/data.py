"""BirdSet data pipeline for SSAMBA training with 10s event-centered windows.

Loads pre-computed mel spectrograms from disk. Supports patch-aware padding masks
for both 16x16 and 128x1 tokenizers.

Training: event-centered 10s windows (all real frames, mask all True)
Validation: 5s clips padded to 10s (mask reflects real vs padded tokens)
"""

import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
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


class CachedMelDataset(Dataset):
    """Dataset that loads pre-computed mel spectrograms from disk.

    Expects files named {idx:06d}.pt containing:
        mel: (1, n_mels, target_frames) float16 tensor
        labels: (num_classes,) float32 tensor
        real_frames: int
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(
            [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f"No cached .pt files in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.cache_dir / self.files[idx], weights_only=True)
        return {
            "mel": data["mel"].float(),
            "labels": data["labels"],
            "real_frames": int(data["real_frames"]),
        }


def _compute_padding_mask(real_frames, target_frames, patch_size, n_mels=128):
    """Compute patch-granularity padding mask.

    For 128x1 patches: each token = 1 time frame, so num_tokens = target_frames.
    For 16x16 patches: tokens are arranged in a 2D grid (f_dim x t_dim), flattened.
        Padding mask is True for tokens whose time position has real frames.

    Args:
        real_frames: Number of real spectrogram frames
        target_frames: Total spectrogram frames (including padding)
        patch_size: (freq_patch, time_patch) tuple
        n_mels: Number of mel frequency bins

    Returns:
        (num_tokens,) boolean tensor, True for real tokens
    """
    f_patch, t_patch = patch_size
    f_dim = n_mels // f_patch
    t_dim = target_frames // t_patch
    num_tokens = f_dim * t_dim

    # How many time-patches are fully real?
    real_t_patches = min(real_frames // t_patch, t_dim)

    # Conv2d flatten(2) produces freq-major order: token i has
    # freq_idx = i // t_dim, time_idx = i % t_dim
    # A token is real if its time position < real_t_patches
    indices = torch.arange(num_tokens)
    mask = (indices % t_dim) < real_t_patches

    return mask


class BirdSetDataModule(pl.LightningDataModule):
    """Lightning DataModule for pre-computed 10s event-centered mel spectrograms."""

    def __init__(
        self,
        dataset_name: str = "HSN",
        batch_size: int = 16,
        num_workers: int = 8,
        target_frames: int = 1000,
        patch_size: tuple = (128, 1),
        n_mels: int = 128,
        cache_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_frames = target_frames
        self.patch_size = tuple(patch_size)
        self.n_mels = n_mels
        self.num_classes = BIRDSET_NUM_CLASSES[dataset_name]
        self.cache_dir = cache_dir

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not self.cache_dir:
                raise ValueError("cache_dir is required for experiment 007")

            train_cache = Path(self.cache_dir) / "train"
            val_cache = Path(self.cache_dir) / "val"
            self.train_dataset = CachedMelDataset(train_cache)
            self.val_dataset = CachedMelDataset(val_cache)
            logger.info(
                f"Using cached mels: {len(self.train_dataset)} train, "
                f"{len(self.val_dataset)} val"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
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
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def _collate(self, batch):
        """Collate with patch-aware padding masks.

        Returns:
            (mel_specs, labels, padding_mask) where:
                mel_specs: (B, 1, n_mels, target_frames)
                labels: (B, num_classes)
                padding_mask: (B, num_tokens) boolean, True for real tokens
        """
        mel_specs = torch.stack([item["mel"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        masks = []
        for item in batch:
            mask = _compute_padding_mask(
                item["real_frames"],
                self.target_frames,
                self.patch_size,
                self.n_mels,
            )
            masks.append(mask)
        padding_mask = torch.stack(masks)

        return mel_specs, labels, padding_mask
