"""Cached-embedding dataset and DataModule for stage 2 training.

Supports two layouts:
  - Single shard: one directory with (embedding.npy, labels.npy, index.parquet, meta.json).
  - Multi-shard (focal): a parent directory containing shard-000/, shard-001/, etc.
    The multi-shard case unions all shards into one logical dataset while keeping
    per-shard memmaps separate (no giant numpy concat, just index math).

Backed by numpy memmap — no audio I/O, no ONNX, just fast tensor loads.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from birdclef_105 import cache as cache_mod

logger = logging.getLogger(__name__)


class CachedEmbeddingDataset(Dataset):
    """Reads a single shard, filtered by split ('train' or 'val').

    Pre-loads embeddings, labels, and optionally spatial features into
    contiguous RAM tensors at init. The full focal set is ~720 MB fp32
    for embeddings + ~50 MB for labels — trivially fits in memory. Spatial
    features add ~12 GB fp32 but are only loaded when `load_spatial=True`.
    """

    def __init__(self, shard_dir: str | Path, split: str, load_spatial: bool = False):
        shard = cache_mod.read_shard(shard_dir)
        index = shard["index"]
        self.species_list = shard["species_list"]
        self.num_classes = len(self.species_list)

        mask = (index["split"] == split).to_numpy()
        rows = np.where(mask)[0]
        self.index_df = index.iloc[rows].reset_index(drop=True)

        # Pre-load into contiguous fp32 tensors — fast __getitem__.
        self._embeddings = torch.from_numpy(
            np.asarray(shard["embeddings"][rows], dtype=np.float32)
        )
        self._labels = torch.from_numpy(
            np.asarray(shard["labels"][rows], dtype=np.float32)
        )
        self._spatial = None
        if load_spatial and shard["spatial"] is not None:
            self._spatial = torch.from_numpy(
                np.asarray(shard["spatial"][rows], dtype=np.float32)
            )

    def __len__(self) -> int:
        return len(self._embeddings)

    def __getitem__(self, i: int) -> dict:
        out = {"embedding": self._embeddings[i], "labels": self._labels[i]}
        if self._spatial is not None:
            out["spatial"] = self._spatial[i]
        return out


class CachedEmbeddingDataModule(pl.LightningDataModule):
    """Single-shard DataModule (soundscape or pilot)."""

    def __init__(self, shard_dir: str | Path, batch_size: int = 256, num_workers: int = 0, load_spatial: bool = False):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_spatial = load_spatial

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = CachedEmbeddingDataset(self.shard_dir, split="train", load_spatial=self.load_spatial)
        self.val_ds = CachedEmbeddingDataset(self.shard_dir, split="val", load_spatial=self.load_spatial)
        self.num_classes = self.train_ds.num_classes
        self.species_list = self.train_ds.species_list

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


def load_sharded_dataset(parent_dir: str | Path, split: str, load_spatial: bool = False) -> ConcatDataset:
    """Load all shard-NNN/ subdirs under parent_dir and union them into one dataset."""
    parent = Path(parent_dir)
    shard_dirs = sorted(parent.glob("shard-*/"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard-*/ subdirs found in {parent}")
    datasets = []
    total = 0
    for sd in shard_dirs:
        ds = CachedEmbeddingDataset(sd, split=split, load_spatial=load_spatial)
        if len(ds) > 0:
            datasets.append(ds)
            total += len(ds)
    if not datasets:
        raise ValueError(f"All shards empty for split={split!r} in {parent}")
    logger.info("Loaded %d shards from %s (split=%s, total=%d)", len(datasets), parent, split, total)
    return ConcatDataset(datasets)


class FocalDataModule(pl.LightningDataModule):
    """Train on focal shards, validate on focal val split.

    Optionally adds soundscape val as a second val dataloader (for monitoring
    val/sc_cmap alongside the primary val/cmap on focal val).
    """

    def __init__(
        self,
        focal_dir: str | Path,
        soundscape_dir: str | Path | None = None,
        batch_size: int = 256,
        num_workers: int = 2,
        load_spatial: bool = False,
    ):
        super().__init__()
        self.focal_dir = Path(focal_dir)
        self.soundscape_dir = Path(soundscape_dir) if soundscape_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_spatial = load_spatial

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = load_sharded_dataset(self.focal_dir, split="train", load_spatial=self.load_spatial)
        self.val_ds = load_sharded_dataset(self.focal_dir, split="val", load_spatial=self.load_spatial)

        # Get species list and num_classes from the first shard.
        first_shard = sorted(self.focal_dir.glob("shard-*/"))[0]
        meta = cache_mod.read_shard(first_shard)
        self.species_list = meta["species_list"]
        self.num_classes = len(self.species_list)

        self.sc_val_ds = None
        if self.soundscape_dir and self.soundscape_dir.exists():
            try:
                self.sc_val_ds = CachedEmbeddingDataset(self.soundscape_dir, split="val", load_spatial=self.load_spatial)
                logger.info("Soundscape val: %d windows", len(self.sc_val_ds))
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Soundscape val not loaded: %s", e)

        logger.info(
            "Focal train=%d  focal val=%d  soundscape val=%s",
            len(self.train_ds),
            len(self.val_ds),
            len(self.sc_val_ds) if self.sc_val_ds else "N/A",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> list[DataLoader] | DataLoader:
        focal_val = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.sc_val_ds is not None:
            sc_val = DataLoader(
                self.sc_val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            return [focal_val, sc_val]
        return focal_val
