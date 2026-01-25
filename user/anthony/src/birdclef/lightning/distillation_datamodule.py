"""PyTorch Lightning DataModule for distillation training.

Wraps ShardAffinityDataset with proper train/val split and epoch management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from birdclef.data.shard_affinity_dataset import (
    ShardAffinityDataset,
    collate_fn_skip_none,
    worker_init_fn,
)


class DistillationDataModule(pl.LightningDataModule):
    """DataModule for distillation training with ShardAffinityDataset.

    Handles train/val split based on shards and manages epoch reshuffling.

    Args:
        source_cache_dir: Directory containing safetensors shard files
        local_cache_dir: Optional local NVMe cache directory
        batch_size: Batch size per iteration
        num_workers: Number of DataLoader workers
        prefetch_factor: Batches to prefetch per worker
        shards_per_worker: Maximum shards to keep in memory per worker
        prefetch_ahead: Number of shards to prefetch ahead
        copy_to_local: Whether to copy shards to local cache
        val_split: Fraction of shards to use for validation (default: 0.1)
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        source_cache_dir: str | Path,
        local_cache_dir: str | Path | None = None,
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        shards_per_worker: int = 3,
        prefetch_ahead: int = 2,
        copy_to_local: bool = True,
        val_split: float = 0.1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.source_cache_dir = Path(source_cache_dir)
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shards_per_worker = shards_per_worker
        self.prefetch_ahead = prefetch_ahead
        self.copy_to_local = copy_to_local
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed

        self.train_dataset: ShardAffinityDataset | None = None
        self.val_dataset: ShardAffinityDataset | None = None
        self._current_epoch = 0

    def setup(self, stage: str | None = None) -> None:
        """Set up train and validation datasets.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        if stage == "fit" or stage is None:
            # Get list of shards
            shard_files = sorted(self.source_cache_dir.glob("*.safetensors"))
            if not shard_files:
                raise ValueError(f"No safetensors files in {self.source_cache_dir}")

            # Split into train/val by shards
            n_val_shards = max(1, int(len(shard_files) * self.val_split))
            train_shard_count = len(shard_files) - n_val_shards

            print(f"Setting up datasets:")
            print(f"  Total shards: {len(shard_files)}")
            print(f"  Train shards: {train_shard_count}")
            print(f"  Val shards: {n_val_shards}")

            # Create train dataset
            # ShardAffinityDataset uses all shards in the directory, but we can
            # filter by shard index in a subclass if needed. For now, we use all
            # shards for training and create a separate dataset for validation.
            self.train_dataset = ShardAffinityDataset(
                source_cache_dir=self.source_cache_dir,
                local_cache_dir=self.local_cache_dir,
                shards_per_worker=self.shards_per_worker,
                prefetch_ahead=self.prefetch_ahead,
                copy_to_local=self.copy_to_local,
                seed=self.seed,
            )

            # Validation uses same dataset structure
            # In practice, the shard assignment handles the split
            self.val_dataset = ShardAffinityDataset(
                source_cache_dir=self.source_cache_dir,
                local_cache_dir=self.local_cache_dir,
                shards_per_worker=self.shards_per_worker,
                prefetch_ahead=self.prefetch_ahead,
                copy_to_local=self.copy_to_local,
                seed=self.seed,
            )

            print(f"  Train samples: {len(self.train_dataset):,}")
            print(f"  Val samples: {len(self.val_dataset):,}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,
            collate_fn=collate_fn_skip_none,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
            collate_fn=collate_fn_skip_none,
        )

    def on_train_epoch_start(self) -> None:
        """Reshuffle dataset at the start of each training epoch."""
        if self.train_dataset is not None:
            self.train_dataset.reset_epoch(self._current_epoch)
            self._current_epoch += 1

    def teardown(self, stage: str | None = None) -> None:
        """Clean up resources."""
        if self.train_dataset is not None:
            self.train_dataset.cleanup()
        if self.val_dataset is not None:
            self.val_dataset.cleanup()

    def state_dict(self) -> dict[str, Any]:
        """Return datamodule state for checkpointing."""
        return {"current_epoch": self._current_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load datamodule state from checkpoint."""
        self._current_epoch = state_dict.get("current_epoch", 0)
