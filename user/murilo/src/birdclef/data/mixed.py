"""Mixed DataModule combining focal recordings and labeled soundscape segments.

Used by Experiment 012 (soundscape fine-tuning) to train on both data sources
simultaneously, with 234-class labels covering all submission species.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from .birdclef2025 import BirdCLEFDataset, build_species_list
from .soundscape import (
    FocalDataset234,
    SoundscapeDataset,
    build_full_species_list,
    load_soundscape_labels,
)

logger = logging.getLogger(__name__)


class MixedDataModule(pl.LightningDataModule):
    """DataModule combining focal recordings and labeled soundscape segments.

    Creates 234-class datasets from both sources:
    - Focal data: uses existing BirdCLEFDataset (206-dim) wrapped in FocalDataset234
    - Soundscape data: uses SoundscapeDataset with multi-label annotations

    Splits:
    - Focal: stratified 80/10/10 by primary_label (same as exp 011)
    - Soundscape: by file (not segment) to prevent data leakage

    Training uses ConcatDataset to combine both sources.
    """

    def __init__(
        self,
        data_dir: str | Path,
        taxonomy_csv: str | Path,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        batch_size: int = 32,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        soundscape_val_files: int = 8,
        soundscape_test_files: int = 8,
        seed: int = 42,
        transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.data_dir = Path(data_dir)
        self.taxonomy_csv = Path(taxonomy_csv)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.soundscape_val_files = soundscape_val_files
        self.soundscape_test_files = soundscape_test_files
        self.seed = seed
        self.transform = transform

        # Build the full 234-species list
        self.species_list = build_full_species_list(taxonomy_csv)
        self.num_classes = len(self.species_list)

        # Build the 206-species list from focal data
        focal_csv = self.data_dir / "train.csv"
        self.species_206 = build_species_list(focal_csv)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _split_focal_dataframe(self, df: pd.DataFrame):
        """Stratified three-way split of focal data by primary_label.

        Same logic as BirdCLEFDataModule._split_dataframe().
        """
        rng = np.random.RandomState(self.seed)
        train_indices = []
        val_indices = []
        test_indices = []

        for _species, group in df.groupby("primary_label"):
            indices = group.index.tolist()
            rng.shuffle(indices)
            n = len(indices)

            if n >= 3:
                n_test = max(1, int(n * self.test_fraction))
                n_val = max(1, int(n * self.val_fraction))
                if n_test + n_val >= n:
                    n_test = 1
                    n_val = 1
                test_indices.extend(indices[:n_test])
                val_indices.extend(indices[n_test : n_test + n_val])
                train_indices.extend(indices[n_test + n_val :])
            elif n == 2:
                train_indices.append(indices[0])
                val_indices.append(indices[1])
                test_indices.append(indices[0])
            else:
                train_indices.append(indices[0])
                val_indices.append(indices[0])
                test_indices.append(indices[0])

        return (
            df.loc[train_indices].reset_index(drop=True),
            df.loc[val_indices].reset_index(drop=True),
            df.loc[test_indices].reset_index(drop=True),
        )

    def _split_soundscape_by_file(self, labels_df: pd.DataFrame):
        """Split soundscape labels by file into train/val/test.

        Ensures all segments from one file go to the same split
        (prevents data leakage from correlated consecutive segments).
        """
        rng = np.random.RandomState(self.seed + 1)  # different seed from focal split

        unique_files = sorted(labels_df["filename"].unique())
        rng.shuffle(unique_files)

        n_files = len(unique_files)
        n_test = min(self.soundscape_test_files, n_files // 4)
        n_val = min(self.soundscape_val_files, n_files // 4)

        test_files = set(unique_files[:n_test])
        val_files = set(unique_files[n_test : n_test + n_val])
        train_files = set(unique_files[n_test + n_val :])

        train_df = labels_df[labels_df["filename"].isin(train_files)].reset_index(drop=True)
        val_df = labels_df[labels_df["filename"].isin(val_files)].reset_index(drop=True)
        test_df = labels_df[labels_df["filename"].isin(test_files)].reset_index(drop=True)

        logger.info(
            f"Soundscape split: {len(train_files)} train files ({len(train_df)} segments), "
            f"{len(val_files)} val files ({len(val_df)} segments), "
            f"{len(test_files)} test files ({len(test_df)} segments)"
        )

        return train_df, val_df, test_df

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        # --- Focal data (206 species, remapped to 234) ---
        focal_csv = self.data_dir / "train.csv"
        focal_audio_dir = self.data_dir / "train_audio"
        focal_df = pd.read_csv(focal_csv)

        focal_train_df, focal_val_df, focal_test_df = self._split_focal_dataframe(focal_df)
        logger.info(
            f"Focal split: {len(focal_train_df)} train, {len(focal_val_df)} val, "
            f"{len(focal_test_df)} test recordings"
        )

        focal_train_206 = BirdCLEFDataset(
            audio_dir=str(focal_audio_dir),
            metadata_df=focal_train_df,
            species_list=self.species_206,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )
        focal_val_206 = BirdCLEFDataset(
            audio_dir=str(focal_audio_dir),
            metadata_df=focal_val_df,
            species_list=self.species_206,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )
        focal_test_206 = BirdCLEFDataset(
            audio_dir=str(focal_audio_dir),
            metadata_df=focal_test_df,
            species_list=self.species_206,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )

        # Wrap focal datasets to remap 206→234
        focal_train = FocalDataset234(focal_train_206, self.species_206, self.species_list)
        focal_val = FocalDataset234(focal_val_206, self.species_206, self.species_list)
        focal_test = FocalDataset234(focal_test_206, self.species_206, self.species_list)

        # --- Soundscape data (234 species) ---
        soundscape_dir = self.data_dir / "train_soundscapes"
        soundscape_labels_csv = self.data_dir / "train_soundscapes_labels.csv"
        soundscape_df = load_soundscape_labels(soundscape_labels_csv)

        sc_train_df, sc_val_df, sc_test_df = self._split_soundscape_by_file(soundscape_df)

        sc_train = SoundscapeDataset(
            soundscape_dir=str(soundscape_dir),
            labels_df=sc_train_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )
        sc_val = SoundscapeDataset(
            soundscape_dir=str(soundscape_dir),
            labels_df=sc_val_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )
        sc_test = SoundscapeDataset(
            soundscape_dir=str(soundscape_dir),
            labels_df=sc_test_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )

        # --- Combine ---
        self.train_dataset = ConcatDataset([focal_train, sc_train])
        self.val_dataset = ConcatDataset([focal_val, sc_val])
        self.test_dataset = ConcatDataset([focal_test, sc_test])

        logger.info(
            f"Combined datasets: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test chunks"
        )
        logger.info(f"  Focal: {len(focal_train)}+{len(focal_val)}+{len(focal_test)} chunks")
        logger.info(f"  Soundscape: {len(sc_train)}+{len(sc_val)}+{len(sc_test)} segments")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
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
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
