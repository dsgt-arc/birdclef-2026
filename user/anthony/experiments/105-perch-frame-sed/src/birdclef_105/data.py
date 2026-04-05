"""Cached-embedding dataset and DataModule for stage 2 training.

Reads a shard written by `cache.write_shard`, filters by the `split` column,
and serves `(embedding, labels)` tensors. Backed by numpy memmap — no audio
I/O, no ONNX, just fast tensor loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from birdclef_105 import cache as cache_mod


class CachedEmbeddingDataset(Dataset):
    def __init__(self, shard_dir: str | Path, split: str):
        shard = cache_mod.read_shard(shard_dir)
        index = shard["index"]
        self.species_list = shard["species_list"]
        self.num_classes = len(self.species_list)

        mask = (index["split"] == split).to_numpy()
        self.rows = np.where(mask)[0]
        if len(self.rows) == 0:
            raise ValueError(f"Split {split!r} is empty in shard {shard_dir}")

        # Keep the full memmaps; we index by row.
        self._embeddings = shard["embeddings"]  # memmap float16 (N, 1536)
        self._labels = shard["labels"]          # memmap int8    (N, num_classes)
        self.index_df = index.iloc[self.rows].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = int(self.rows[i])
        emb = torch.from_numpy(np.asarray(self._embeddings[row], dtype=np.float32))
        lab = torch.from_numpy(np.asarray(self._labels[row], dtype=np.float32))
        return {"embedding": emb, "labels": lab}


class CachedEmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        shard_dir: str | Path,
        batch_size: int = 256,
        num_workers: int = 2,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = CachedEmbeddingDataset(self.shard_dir, split="train")
        self.val_ds = CachedEmbeddingDataset(self.shard_dir, split="val")
        self.num_classes = self.train_ds.num_classes
        self.species_list = self.train_ds.species_list

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
