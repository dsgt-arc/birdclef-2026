"""Cached-feature datasets for stage 2 training.

Two dataset types:
  - CachedEmbeddingDataset (map-style): memmap-backed, for val and small shards.
    Sequential access (no shuffle) works well with memmap.
  - StreamingShardDataset (iterable): loads one shard at a time into RAM,
    shuffles within, then GCs before loading the next. Peak RAM = one shard
    (~6 GB spatial fp16). Used for training over 8 focal shards without
    preloading 50 GB.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset

from birdclef_105 import cache as cache_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Map-style dataset (memmap, for val / small data)
# ---------------------------------------------------------------------------

class CachedEmbeddingDataset(Dataset):
    """Reads a single shard filtered by split. Memmap-backed, zero preload.

    Best for val (sequential, no shuffle) or small datasets (soundscape/pilot).
    For shuffled training over many large shards, use StreamingShardDataset.
    """

    def __init__(self, shard_dir: str | Path, split: str, load_spatial: bool = False):
        shard = cache_mod.read_shard(shard_dir)
        index = shard["index"]
        self.species_list = shard["species_list"]
        self.num_classes = len(self.species_list)

        mask = (index["split"] == split).to_numpy()
        self.rows = np.where(mask)[0]
        self.index_df = index.iloc[self.rows].reset_index(drop=True)

        self._embeddings = shard["embeddings"]  # memmap fp16
        self._labels = shard["labels"]          # memmap int8
        self._spatial = shard["spatial"] if load_spatial else None  # memmap fp16 or None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = int(self.rows[i])
        out = {
            "embedding": torch.from_numpy(np.asarray(self._embeddings[row], dtype=np.float32)),
            "labels": torch.from_numpy(np.asarray(self._labels[row], dtype=np.float32)),
        }
        if self._spatial is not None:
            out["spatial"] = torch.from_numpy(np.asarray(self._spatial[row], dtype=np.float32))
        return out


# ---------------------------------------------------------------------------
# Streaming dataset (one shard at a time, for large-scale training)
# ---------------------------------------------------------------------------

class StreamingShardDataset(IterableDataset):
    """Streams focal shards one at a time. Peak RAM = one shard.

    Each epoch:
      1. Shuffle shard order
      2. For each shard: load into RAM, shuffle rows, yield samples, free
    """

    def __init__(self, shard_dirs: list[Path], split: str, load_spatial: bool = False):
        self.shard_dirs = list(shard_dirs)
        self.split = split
        self.load_spatial = load_spatial
        # Pre-compute total length from index parquets (cheap, no data loaded).
        self._total = 0
        for sd in self.shard_dirs:
            idx = pd.read_parquet(Path(sd) / "index.parquet")
            self._total += int((idx["split"] == split).sum())

    def __len__(self) -> int:
        """Hint for Lightning progress bar. Exact count."""
        return self._total

    def __iter__(self):
        shards = list(self.shard_dirs)
        random.shuffle(shards)
        for shard_dir in shards:
            yield from self._stream_shard(shard_dir)

    def _stream_shard(self, shard_dir: Path):
        """Load one shard into RAM, yield shuffled samples, then release."""
        shard = cache_mod.read_shard(shard_dir)
        index = shard["index"]
        mask = (index["split"] == self.split).to_numpy()
        rows = np.where(mask)[0]
        if len(rows) == 0:
            return

        # Load this shard's data into contiguous arrays (one shard fits in RAM).
        emb = np.asarray(shard["embeddings"][rows], dtype=np.float32)   # ~90 MB
        lab = np.asarray(shard["labels"][rows], dtype=np.float32)       # ~2.5 MB
        spat = None
        if self.load_spatial and shard["spatial"] is not None:
            spat = np.asarray(shard["spatial"][rows], dtype=np.float16)  # ~5.6 GB fp16

        # Shuffle within shard.
        perm = np.random.permutation(len(rows))

        for i in perm:
            out = {
                "embedding": torch.from_numpy(emb[i].copy()),
                "labels": torch.from_numpy(lab[i].copy()),
            }
            if spat is not None:
                out["spatial"] = torch.from_numpy(spat[i].astype(np.float32))
            yield out

        # Explicit cleanup before next shard.
        del emb, lab, spat


# ---------------------------------------------------------------------------
# DataModules
# ---------------------------------------------------------------------------

class CachedEmbeddingDataModule(pl.LightningDataModule):
    """Single-shard DataModule (soundscape or pilot)."""

    def __init__(self, shard_dir: str | Path, batch_size: int = 256, num_workers: int = 0, load_spatial: bool = False):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_spatial = load_spatial
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return
        self.train_ds = CachedEmbeddingDataset(self.shard_dir, split="train", load_spatial=self.load_spatial)
        self.val_ds = CachedEmbeddingDataset(self.shard_dir, split="val", load_spatial=self.load_spatial)
        self.num_classes = self.train_ds.num_classes
        self.species_list = self.train_ds.species_list
        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )


class FocalDataModule(pl.LightningDataModule):
    """Train on focal shards (streaming), val on focal val + soundscape val.

    Training uses StreamingShardDataset — loads one shard at a time (~6 GB peak).
    Val uses CachedEmbeddingDataset (memmap, sequential access = fine).
    """

    def __init__(
        self,
        focal_dir: str | Path,
        soundscape_dir: str | Path | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        load_spatial: bool = False,
    ):
        super().__init__()
        self.focal_dir = Path(focal_dir)
        self.soundscape_dir = Path(soundscape_dir) if soundscape_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_spatial = load_spatial
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        shard_dirs = sorted(self.focal_dir.glob("shard-*/"))
        if not shard_dirs:
            raise FileNotFoundError(f"No shard-*/ subdirs in {self.focal_dir}")

        # Train: streaming (one shard at a time).
        self.train_ds = StreamingShardDataset(
            shard_dirs=shard_dirs,
            split="train",
            load_spatial=self.load_spatial,
        )

        # Val: memmap (sequential access, no shuffle).
        val_datasets = []
        for sd in shard_dirs:
            ds = CachedEmbeddingDataset(sd, split="val", load_spatial=self.load_spatial)
            if len(ds) > 0:
                val_datasets.append(ds)
        self.val_ds = ConcatDataset(val_datasets) if val_datasets else None

        # Species list from first shard.
        meta = cache_mod.read_shard(shard_dirs[0])
        self.species_list = meta["species_list"]
        self.num_classes = len(self.species_list)

        # Soundscape auxiliary val.
        self.sc_val_ds = None
        if self.soundscape_dir and self.soundscape_dir.exists():
            try:
                self.sc_val_ds = CachedEmbeddingDataset(
                    self.soundscape_dir, split="val", load_spatial=self.load_spatial
                )
                logger.info("Soundscape val: %d windows", len(self.sc_val_ds))
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Soundscape val not loaded: %s", e)

        logger.info(
            "Focal train=%d (streaming)  focal val=%d (memmap)  soundscape val=%s",
            len(self.train_ds),
            len(self.val_ds) if self.val_ds else 0,
            len(self.sc_val_ds) if self.sc_val_ds else "N/A",
        )
        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,  # IterableDataset handles its own shuffling
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> list[DataLoader] | DataLoader:
        focal_val = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        if self.sc_val_ds is not None:
            sc_val = DataLoader(
                self.sc_val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
            return [focal_val, sc_val]
        return focal_val
