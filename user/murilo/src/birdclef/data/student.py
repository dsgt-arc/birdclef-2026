"""Student training data module with pseudo-labeled soundscape support.

Used by Experiment 015 to train on three data sources:
  1. Focal recordings (hard labels, 206→234 remapped)
  2. Labeled soundscapes (hard labels, 234-dim)
  3. Pseudo-labeled soundscapes (soft labels from exp 014, 234-dim)

The pseudo-labeled data is only included in the training set.
Validation and test sets use only ground-truth labeled data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .birdclef2025 import BirdCLEFDataset, _load_audio_chunk, build_species_list
from .soundscape import (
    FocalDataset234,
    SoundscapeDataset,
    build_full_species_list,
    load_soundscape_labels,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Split helpers (copied from MixedDataModule to avoid modifying exp 012 code)
# ============================================================================


def _split_focal_dataframe(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified three-way split of focal data by primary_label."""
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for _species, group in df.groupby("primary_label"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n = len(indices)

        if n >= 3:
            n_test = max(1, int(n * test_fraction))
            n_val = max(1, int(n * val_fraction))
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


def _split_soundscape_by_file(
    labels_df: pd.DataFrame,
    soundscape_val_files: int = 8,
    soundscape_test_files: int = 8,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split soundscape labels by file into train/val/test."""
    rng = np.random.RandomState(seed)

    unique_files = sorted(labels_df["filename"].unique())
    rng.shuffle(unique_files)

    n_files = len(unique_files)
    n_test = min(soundscape_test_files, n_files // 4)
    n_val = min(soundscape_val_files, n_files // 4)

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


# ============================================================================
# Dataset classes
# ============================================================================


class WeightedDatasetWrapper(Dataset):
    """Wraps a (waveform, labels) dataset to return (waveform, labels, weight).

    Allows mixing datasets with different per-sample loss weights in a
    ConcatDataset. Ground-truth labeled data uses weight=1.0; pseudo-labeled
    data uses a lower weight to control its influence on the loss.
    """

    def __init__(self, dataset: Dataset, weight: float = 1.0):
        self.dataset = dataset
        self.weight = weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, labels = self.dataset[idx]
        return waveform, labels, self.weight


class PseudoLabeledDataset(Dataset):
    """Dataset for pseudo-labeled soundscape segments from exp 014.

    Loads soft label probabilities from parquet files and audio from
    train_soundscapes/. Each sample returns (waveform, soft_labels, weight).

    Args:
        parquet_dir: Directory containing pseudo_labels*.parquet shard files.
        soundscape_dir: Directory containing soundscape .ogg audio files.
        species_list: Sorted list of 234 species (must match parquet columns).
        sample_rate: Target sample rate for audio loading.
        chunk_duration: Duration of each segment in seconds.
        weight: Per-sample loss weight for pseudo-labeled data.
        transform: Optional audio transform (e.g., mel spectrogram).
    """

    def __init__(
        self,
        parquet_dir: str | Path,
        soundscape_dir: str | Path,
        species_list: list[str],
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        weight: float = 0.5,
        transform=None,
    ):
        self.soundscape_dir = Path(soundscape_dir)
        self.species_list = species_list
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.weight = weight
        self.transform = transform

        # Load and concatenate all parquet shards
        parquet_dir = Path(parquet_dir)
        parquet_files = sorted(parquet_dir.glob("pseudo_labels*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No pseudo_labels*.parquet files in {parquet_dir}")

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        df = df.sort_values(["filename", "end_sec"]).reset_index(drop=True)

        # Verify species columns match
        species_cols = [c for c in df.columns if c not in ("filename", "end_sec")]
        if species_cols != list(species_list):
            logger.warning(
                f"Parquet species columns ({len(species_cols)}) differ from "
                f"species_list ({len(species_list)}). Using parquet column order."
            )

        # Pre-extract arrays for fast __getitem__
        self.filenames = df["filename"].tolist()
        self.end_secs = df["end_sec"].values.astype(np.float64)
        self.labels_matrix = df[species_cols].values.astype(np.float32)

        # Verify audio files exist (sample check)
        unique_files = set(self.filenames)
        missing = [f for f in list(unique_files)[:10] if not (self.soundscape_dir / f).exists()]
        if missing:
            logger.warning(f"Missing audio files (first {len(missing)}): {missing}")

        logger.info(
            f"PseudoLabeledDataset: {len(self)} segments from {len(unique_files)} files "
            f"({len(species_cols)} species, weight={weight})"
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        end_sec = self.end_secs[idx]
        start_sec = end_sec - self.chunk_duration

        audio_path = self.soundscape_dir / filename
        start_sample = int(start_sec * self.sample_rate)

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=start_sample,
            num_samples=self.chunk_samples,
        )

        # Pad to exact chunk length if needed
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        if self.transform is not None:
            waveform = self.transform(waveform)

        soft_labels = torch.from_numpy(self.labels_matrix[idx])
        return waveform, soft_labels, self.weight


# ============================================================================
# DataModule
# ============================================================================


class StudentDataModule(pl.LightningDataModule):
    """DataModule combining focal, labeled soundscape, and pseudo-labeled data.

    Extends the MixedDataModule pattern from exp 012 with a third data source:
    pseudo-labeled soundscape segments from exp 014. All datasets return
    (waveform, labels, weight) tuples for per-sample loss weighting.

    Training set: focal (w=1.0) + labeled soundscape (w=1.0) + pseudo-labeled (w=configurable)
    Val/test sets: focal (w=1.0) + labeled soundscape (w=1.0) only
    """

    def __init__(
        self,
        data_dir: str | Path,
        taxonomy_csv: str | Path,
        pseudo_label_dir: str | Path,
        pseudo_label_weight: float = 0.5,
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
        self.pseudo_label_dir = Path(pseudo_label_dir)
        self.pseudo_label_weight = pseudo_label_weight
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

        # Build species lists
        self.species_list = build_full_species_list(taxonomy_csv)
        self.num_classes = len(self.species_list)
        self.species_206 = build_species_list(self.data_dir / "train.csv")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        # --- Focal data (206 species, remapped to 234) ---
        focal_csv = self.data_dir / "train.csv"
        focal_audio_dir = self.data_dir / "train_audio"
        focal_df = pd.read_csv(focal_csv)

        focal_train_df, focal_val_df, focal_test_df = _split_focal_dataframe(
            focal_df, self.val_fraction, self.test_fraction, self.seed
        )
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

        # Remap 206→234
        focal_train = FocalDataset234(focal_train_206, self.species_206, self.species_list)
        focal_val = FocalDataset234(focal_val_206, self.species_206, self.species_list)
        focal_test = FocalDataset234(focal_test_206, self.species_206, self.species_list)

        # --- Labeled soundscape data (234 species) ---
        soundscape_dir = self.data_dir / "train_soundscapes"
        soundscape_labels_csv = self.data_dir / "train_soundscapes_labels.csv"
        soundscape_df = load_soundscape_labels(soundscape_labels_csv)

        sc_train_df, sc_val_df, sc_test_df = _split_soundscape_by_file(
            soundscape_df,
            self.soundscape_val_files,
            self.soundscape_test_files,
            self.seed + 1,
        )

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

        # --- Pseudo-labeled soundscape data (train only) ---
        pseudo_labeled = PseudoLabeledDataset(
            parquet_dir=str(self.pseudo_label_dir),
            soundscape_dir=str(soundscape_dir),
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            weight=self.pseudo_label_weight,
            transform=self.transform,
        )

        # --- Wrap labeled datasets with weight=1.0 ---
        wrapped_focal_train = WeightedDatasetWrapper(focal_train, weight=1.0)
        wrapped_focal_val = WeightedDatasetWrapper(focal_val, weight=1.0)
        wrapped_focal_test = WeightedDatasetWrapper(focal_test, weight=1.0)

        wrapped_sc_train = WeightedDatasetWrapper(sc_train, weight=1.0)
        wrapped_sc_val = WeightedDatasetWrapper(sc_val, weight=1.0)
        wrapped_sc_test = WeightedDatasetWrapper(sc_test, weight=1.0)

        # --- Combine: train includes pseudo-labels, val/test do not ---
        self.train_dataset = ConcatDataset([wrapped_focal_train, wrapped_sc_train, pseudo_labeled])
        self.val_dataset = ConcatDataset([wrapped_focal_val, wrapped_sc_val])
        self.test_dataset = ConcatDataset([wrapped_focal_test, wrapped_sc_test])

        logger.info(
            f"Student datasets: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test"
        )
        logger.info(f"  Focal: {len(focal_train)}+{len(focal_val)}+{len(focal_test)}")
        logger.info(f"  Soundscape: {len(sc_train)}+{len(sc_val)}+{len(sc_test)}")
        logger.info(f"  Pseudo-labeled: {len(pseudo_labeled)} (weight={self.pseudo_label_weight})")

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
