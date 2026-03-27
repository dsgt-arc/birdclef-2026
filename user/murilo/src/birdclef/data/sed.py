"""SED data module with 10-second windows for Sound Event Detection training.

Provides datasets that load 10-second audio windows instead of the standard
5-second chunks. Focal recordings use random 10s crops; soundscape segments
use 10s centered on the labeled 5s window (2.5s context on each side).

Used by Experiment 016 (SED architecture).
"""

import logging
import random as py_random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .dataset import _load_audio_chunk, build_species_list
from .soundscape import (
    FocalDataset234,
    build_full_species_list,
    load_soundscape_labels,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Split helpers (same logic as MixedDataModule / StudentDataModule)
# ============================================================================


def _split_focal_dataframe(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split of focal data by primary_label.

    Supports three modes:
    - 80/10/10 (test_fraction>0, val_fraction>0): train/val/test split
    - 90/10 (test_fraction=0, val_fraction>0): train/val only
    - 100/0/0 (val_fraction=0): fulldata training, all samples in train
    """
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    has_test = test_fraction > 0
    has_val = val_fraction > 0

    # Fulldata mode: everything goes to train
    if not has_val:
        return df, df.iloc[:0], df.iloc[:0]

    for _species, group in df.groupby("primary_label"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n = len(indices)

        if has_test:
            # Three-way split (original 80/10/10 behavior)
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
        else:
            # Two-way split (90/10 train/val only)
            if n >= 2:
                n_val = max(1, int(n * val_fraction))
                if n_val >= n:
                    n_val = 1
                val_indices.extend(indices[:n_val])
                train_indices.extend(indices[n_val:])
            else:
                # Single sample: must appear in both for species coverage
                train_indices.append(indices[0])
                val_indices.append(indices[0])

    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True)
        if test_indices
        else df.iloc[:0].reset_index(drop=True),
    )


def _split_soundscape_by_file(
    labels_df: pd.DataFrame,
    soundscape_val_files: int = 8,
    soundscape_test_files: int = 8,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split soundscape labels by file into train/val/test.

    When soundscape_test_files=0, performs a two-way split (train/val only).
    Returns an empty test DataFrame to preserve the 3-tuple interface.
    """
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


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS timestamp to seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


class SEDFocalDataset(Dataset):
    """Focal recordings with random 10s crops for SED training.

    Unlike BirdCLEFDataset which pre-chunks into fixed 5s windows, this dataset
    stores per-recording entries and randomly crops a window at training time.
    Recordings shorter than window_duration are zero-padded.

    Returns:
        (waveform_or_features, labels) — labels is (234,) multi-hot float tensor.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        metadata_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        window_duration: float = 10.0,
        transform=None,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_samples = int(window_duration * sample_rate)
        self.transform = transform

        # Store per-recording metadata (not pre-chunked)
        self.recordings = []
        skipped = 0

        for _, row in metadata_df.iterrows():
            audio_path = self.audio_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            try:
                info = sf.info(str(audio_path))
                total_samples = int(info.frames * sample_rate / info.samplerate)
            except Exception:
                skipped += 1
                continue

            label_vec = self._build_label_vec(row)
            self.recordings.append((str(audio_path), total_samples, label_vec))

        if skipped > 0:
            logger.warning(f"Skipped {skipped} recordings (file not found or unreadable)")
        logger.info(
            f"SEDFocalDataset: {len(self.recordings)} recordings "
            f"({self.num_classes} classes, {window_duration}s windows)"
        )

    def _build_label_vec(self, row):
        """Build multi-hot label vector from metadata row."""
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        primary = row.get("primary_label", "")
        if primary in self.species_to_idx:
            label_vec[self.species_to_idx[primary]] = 1.0

        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary.strip() and secondary != "[]":
            secondary = secondary.strip("[]' ")
            for s in secondary.replace("'", "").replace('"', "").split(","):
                s = s.strip()
                if s in self.species_to_idx:
                    label_vec[self.species_to_idx[s]] = 1.0

        return label_vec

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        audio_path, total_samples, label_vec = self.recordings[idx]

        # Random start position for the window
        if total_samples > self.window_samples:
            start_sample = py_random.randint(0, total_samples - self.window_samples)
        else:
            start_sample = 0

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=start_sample,
            num_samples=self.window_samples,
        )

        # Pad if recording is shorter than window
        if waveform.shape[0] < self.window_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.window_samples - waveform.shape[0])
            )

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label_vec.clone()


class SEDSoundscapeDataset(Dataset):
    """Soundscape segments with 10s context window for SED training.

    Loads 10s of audio centered on each labeled 5s segment (2.5s context
    before and after). Edge segments are zero-padded when insufficient
    audio context is available.

    Returns:
        (waveform_or_features, labels) — labels is (234,) multi-hot float tensor.
    """

    def __init__(
        self,
        soundscape_dir: str | Path,
        labels_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        target_duration: float = 5.0,
        context_duration: float = 10.0,
        transform=None,
    ):
        self.soundscape_dir = Path(soundscape_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.context_duration = context_duration
        self.context_samples = int(context_duration * sample_rate)
        self.extra_context_samples = int((context_duration - target_duration) / 2 * sample_rate)
        self.transform = transform

        # Build chunk index
        self.chunks = []
        skipped = 0

        for _, row in labels_df.iterrows():
            audio_path = self.soundscape_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            start_sec = _parse_timestamp(row["start"])
            start_sample = int(start_sec * sample_rate)

            # Build multi-hot label vector
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for species_id in str(row["primary_label"]).split(";"):
                species_id = species_id.strip()
                if species_id in self.species_to_idx:
                    label_vec[self.species_to_idx[species_id]] = 1.0

            self.chunks.append((str(audio_path), start_sample, label_vec))

        if skipped > 0:
            logger.warning(f"Skipped {skipped} segments (audio file not found)")
        logger.info(
            f"SEDSoundscapeDataset: {len(self.chunks)} segments from "
            f"{labels_df['filename'].nunique()} files "
            f"({self.num_classes} classes, {context_duration}s context)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, segment_start_sample, label_vec = self.chunks[idx]

        # Load 10s centered on the 5s segment (2.5s before + 2.5s after)
        load_start = max(0, segment_start_sample - self.extra_context_samples)

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=load_start,
            num_samples=self.context_samples,
        )

        # Zero-pad if at file boundary
        if waveform.shape[0] < self.context_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.context_samples - waveform.shape[0])
            )

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label_vec.clone()


# ============================================================================
# DataModule
# ============================================================================


class SEDDataModule(pl.LightningDataModule):
    """DataModule for SED training with 10-second windows.

    Combines focal recordings (random 10s crops) and labeled soundscape
    segments (10s context windows centered on 5s targets).

    Training set: focal (234-dim) + soundscape (234-dim)
    Val/test sets: focal (234-dim) + soundscape (234-dim)
    """

    def __init__(
        self,
        data_dir: str | Path,
        taxonomy_csv: str | Path,
        sample_rate: int = 32000,
        window_duration: float = 10.0,
        batch_size: int = 64,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        soundscape_val_files: int = 8,
        soundscape_test_files: int = 8,
        seed: int = 42,
        transform=None,
        eval_transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform", "eval_transform"])
        self.data_dir = Path(data_dir)
        self.taxonomy_csv = Path(taxonomy_csv)
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.soundscape_val_files = soundscape_val_files
        self.soundscape_test_files = soundscape_test_files
        self.seed = seed
        self.transform = transform
        # eval_transform: separate transform for val/test datasets.
        # When None (default), uses the same transform for all splits.
        # Use this when training on raw waveforms but val/test need mel spectrograms.
        self.eval_transform = eval_transform if eval_transform is not None else transform

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

        has_test = self.test_fraction > 0
        has_val = self.val_fraction > 0
        fulldata = not has_val

        focal_train_df, focal_val_df, focal_test_df = _split_focal_dataframe(
            focal_df, self.val_fraction, self.test_fraction, self.seed
        )
        if fulldata:
            logger.info(f"Focal fulldata: {len(focal_train_df)} train (100%, no val/test)")
        else:
            logger.info(
                f"Focal split: {len(focal_train_df)} train, {len(focal_val_df)} val"
                + (f", {len(focal_test_df)} test" if has_test else " (no test holdout)")
            )

        focal_train_206 = SEDFocalDataset(
            audio_dir=str(focal_audio_dir),
            metadata_df=focal_train_df,
            species_list=self.species_206,
            sample_rate=self.sample_rate,
            window_duration=self.window_duration,
            transform=self.transform,
        )
        # Remap 206→234
        focal_train = FocalDataset234(focal_train_206, self.species_206, self.species_list)

        focal_val = None
        if not fulldata:
            focal_val_206 = SEDFocalDataset(
                audio_dir=str(focal_audio_dir),
                metadata_df=focal_val_df,
                species_list=self.species_206,
                sample_rate=self.sample_rate,
                window_duration=self.window_duration,
                transform=self.eval_transform,
            )
            focal_val = FocalDataset234(focal_val_206, self.species_206, self.species_list)

        if has_test:
            focal_test_206 = SEDFocalDataset(
                audio_dir=str(focal_audio_dir),
                metadata_df=focal_test_df,
                species_list=self.species_206,
                sample_rate=self.sample_rate,
                window_duration=self.window_duration,
                transform=self.eval_transform,
            )
            focal_test = FocalDataset234(focal_test_206, self.species_206, self.species_list)

        # --- Labeled soundscape data (234 species, 10s context) ---
        soundscape_dir = self.data_dir / "train_soundscapes"
        soundscape_labels_csv = self.data_dir / "train_soundscapes_labels.csv"
        soundscape_df = load_soundscape_labels(soundscape_labels_csv)

        soundscape_val_files = 0 if fulldata else self.soundscape_val_files
        soundscape_test_files = self.soundscape_test_files if has_test else 0
        sc_train_df, sc_val_df, sc_test_df = _split_soundscape_by_file(
            soundscape_df,
            soundscape_val_files,
            soundscape_test_files,
            self.seed + 1,
        )

        sc_train = SEDSoundscapeDataset(
            soundscape_dir=str(soundscape_dir),
            labels_df=sc_train_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            target_duration=5.0,
            context_duration=self.window_duration,
            transform=self.transform,
        )

        # --- Combine ---
        if fulldata:
            self.train_dataset = ConcatDataset([focal_train, sc_train])
            self.val_dataset = None
        else:
            sc_val = SEDSoundscapeDataset(
                soundscape_dir=str(soundscape_dir),
                labels_df=sc_val_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                target_duration=5.0,
                context_duration=self.window_duration,
                transform=self.eval_transform,
            )
            self.train_dataset = ConcatDataset([focal_train, sc_train])
            self.val_dataset = ConcatDataset([focal_val, sc_val])

        if has_test:
            sc_test = SEDSoundscapeDataset(
                soundscape_dir=str(soundscape_dir),
                labels_df=sc_test_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                target_duration=5.0,
                context_duration=self.window_duration,
                transform=self.eval_transform,
            )
            self.test_dataset = ConcatDataset([focal_test, sc_test])
        else:
            self.test_dataset = None

        if fulldata:
            logger.info(f"SED datasets: {len(self.train_dataset)} train (fulldata, no val/test)")
            logger.info(f"  Focal: {len(focal_train)}")
        else:
            logger.info(
                f"SED datasets: {len(self.train_dataset)} train, "
                f"{len(self.val_dataset)} val"
                + (f", {len(self.test_dataset)} test" if self.test_dataset else " (no test)")
            )
            logger.info(
                f"  Focal: {len(focal_train)}+{len(focal_val)}"
                + (f"+{len(focal_test)}" if has_test else "")
            )
        logger.info(
            f"  Soundscape: {len(sc_train)}+{len(sc_val)}"
            + (f"+{len(sc_test)}" if has_test else "")
        )

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
        if self.val_dataset is None:
            return None
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
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
