"""Soft-label dataset: loads fused parquets from exp 013, averages 1s tokens into 5s chunks."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader, Dataset

from birdclef_009.data.audio import collate_finetune, load_audio_chunk
from birdclef_009.data.finetune import BirdCLEFTrainDataset, build_species_list

logger = logging.getLogger(__name__)


class SoftLabelDataset(Dataset):
    """Load audio chunks with soft labels from fused parquets.

    Soft labels are at 1s resolution. For each 5s audio chunk, average
    the soft labels from the overlapping 1s tokens.

    Args:
        audio_dir: Root directory containing audio files (species/recording.ogg)
        soft_labels_df: polars DataFrame with columns [file, start_time, ...species...]
        species_list: Ordered list of species codes (must match parquet column order)
        sample_rate: Audio sample rate
        chunk_duration: Duration of each chunk in seconds
        num_tokens: Number of mel tokens per chunk (for padding mask)
    """

    def __init__(
        self,
        audio_dir: str,
        soft_labels_df: pl.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        num_tokens: int = 625,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.num_tokens = num_tokens

        # Build chunk index: for each file, group 1s tokens into chunk_duration chunks
        self.chunks = []
        skipped_files = 0

        # Group by file
        grouped = soft_labels_df.group_by("file", maintain_order=True)

        for file_key, file_df in grouped:
            file_key = file_key[0] if isinstance(file_key, tuple) else file_key
            audio_path = self.audio_dir / file_key
            if not audio_path.exists():
                skipped_files += 1
                continue

            # Sort by start_time
            file_df = file_df.sort("start_time")
            start_times = file_df["start_time"].to_numpy()
            # Extract species label matrix for this file
            label_matrix = file_df.select(species_list).to_numpy()  # (num_tokens_in_file, num_classes)

            # Get file duration from max start_time + 1s (each token is 1s)
            file_duration = start_times[-1] + 1.0

            # Split into chunk_duration chunks
            chunk_start = 0.0
            while chunk_start < file_duration:
                chunk_end = chunk_start + chunk_duration

                # Find 1s tokens that overlap with this chunk
                mask = (start_times >= chunk_start) & (start_times < chunk_end)
                if mask.any():
                    # Average soft labels across overlapping tokens
                    chunk_labels = label_matrix[mask].mean(axis=0).astype(np.float32)
                else:
                    chunk_labels = np.zeros(self.num_classes, dtype=np.float32)

                # Compute real samples (clamp to file boundary)
                real_duration = min(chunk_duration, file_duration - chunk_start)
                real_samples = int(real_duration * sample_rate)

                start_sample = int(chunk_start * sample_rate)
                self.chunks.append((
                    str(audio_path),
                    start_sample,
                    real_samples,
                    torch.from_numpy(chunk_labels),
                ))

                chunk_start += chunk_duration

        if skipped_files > 0:
            logger.warning(f"Skipped {skipped_files} recordings (file not found)")
        logger.info(
            f"SoftLabelDataset: {len(self.chunks)} chunks from "
            f"{soft_labels_df['file'].n_unique()} recordings ({self.num_classes} classes)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, real_samples, label_vec = self.chunks[idx]

        waveform = load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=start_sample, num_samples=real_samples,
        )

        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        # Compute real tokens for padding mask
        hop_samples = self.chunk_samples // self.num_tokens
        real_tokens = min(
            (real_samples + hop_samples - 1) // hop_samples,
            self.num_tokens,
        )

        return {
            "waveform": waveform,
            "clip_labels": label_vec.clone(),
            "num_tokens": self.num_tokens,
            "real_tokens": int(real_tokens),
        }


class SoftLabelDataModule(ptl.LightningDataModule):
    """DataModule that uses soft labels for train, hard labels for val.

    Train: SoftLabelDataset with fused parquet soft labels
    Val: BirdCLEFTrainDataset with hard labels from train.csv (same split as exp 012)

    Args:
        audio_dir: Root audio directory (focal recordings)
        metadata_csv: train.csv path (for val split and species list)
        soft_labels_dir: Directory containing fused-v2 parquets
        species_list: Ordered species list (if None, built from metadata_csv)
        include_soundscape: Whether to include soundscape training data
        soundscape_audio_dir: Root audio directory for soundscapes
        sample_rate: Audio sample rate
        chunk_duration: Chunk duration in seconds
        num_tokens: Number of mel tokens per chunk
        batch_size: Batch size
        num_workers: DataLoader workers
        val_fraction: Fraction for validation split
        seed: Random seed for reproducible split
    """

    def __init__(
        self,
        audio_dir: str,
        metadata_csv: str,
        soft_labels_dir: str,
        species_list: list[str] | None = None,
        include_soundscape: bool = False,
        soundscape_audio_dir: str | None = None,
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        num_tokens: int = 625,
        batch_size: int = 256,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["species_list"])
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.soft_labels_dir = Path(soft_labels_dir).expanduser()
        self.include_soundscape = include_soundscape
        self.soundscape_audio_dir = soundscape_audio_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

        if species_list is not None:
            self.species_list = species_list
        else:
            self.species_list = build_species_list(metadata_csv)

        self.num_classes = len(self.species_list)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # --- Val split: same stratified split as exp 012 ---
            df = pd.read_csv(self.metadata_csv)
            rng = np.random.RandomState(self.seed)
            val_indices = []
            train_indices = []
            for species, group in df.groupby("primary_label"):
                indices = group.index.tolist()
                rng.shuffle(indices)
                n_val = max(1, int(len(indices) * self.val_fraction))
                val_indices.extend(indices[:n_val])
                train_indices.extend(indices[n_val:])

            train_df = df.loc[train_indices].reset_index(drop=True)
            val_df = df.loc[val_indices].reset_index(drop=True)
            logger.info(f"Split: {len(train_df)} train recordings, {len(val_df)} val recordings")

            # --- Val dataset: hard labels (same as exp 012) ---
            self.val_dataset = BirdCLEFTrainDataset(
                audio_dir=self.audio_dir,
                metadata_df=val_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                num_tokens=self.num_tokens,
            )

            # --- Train dataset: soft labels ---
            # Load focal soft labels
            focal_path = self.soft_labels_dir / "focal" / "soft_labels.parquet"
            logger.info(f"Loading focal soft labels from {focal_path}")
            focal_df = pl.read_parquet(focal_path)
            logger.info(f"Focal soft labels: {focal_df.shape}")

            # Filter to train-only recordings (exclude val recordings)
            val_filenames = set(val_df["filename"].tolist())
            focal_df = focal_df.filter(~pl.col("file").is_in(val_filenames))
            logger.info(f"Focal after excluding val: {focal_df.shape}")

            # Optionally add soundscape data
            if self.include_soundscape:
                sc_path = self.soft_labels_dir / "soundscape" / "soft_labels.parquet"
                logger.info(f"Loading soundscape soft labels from {sc_path}")
                sc_df = pl.read_parquet(sc_path)
                logger.info(f"Soundscape soft labels: {sc_df.shape}")

                # Soundscape files need different audio_dir prefix
                # Prepend a marker so SoftLabelDataset can resolve paths
                sc_df = sc_df.with_columns(
                    pl.lit("__soundscape__/").cast(pl.Utf8) + pl.col("file")
                )
                focal_df = focal_df.with_columns(
                    pl.lit("__focal__/").cast(pl.Utf8) + pl.col("file")
                )
                combined_df = pl.concat([focal_df, sc_df])
                logger.info(f"Combined train soft labels: {combined_df.shape}")

                self.train_dataset = _MultiDirSoftLabelDataset(
                    focal_audio_dir=self.audio_dir,
                    soundscape_audio_dir=self.soundscape_audio_dir,
                    soft_labels_df=combined_df,
                    species_list=self.species_list,
                    sample_rate=self.sample_rate,
                    chunk_duration=self.chunk_duration,
                    num_tokens=self.num_tokens,
                )
            else:
                self.train_dataset = SoftLabelDataset(
                    audio_dir=self.audio_dir,
                    soft_labels_df=focal_df,
                    species_list=self.species_list,
                    sample_rate=self.sample_rate,
                    chunk_duration=self.chunk_duration,
                    num_tokens=self.num_tokens,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_finetune,
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
            collate_fn=collate_finetune,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


class _MultiDirSoftLabelDataset(Dataset):
    """SoftLabelDataset variant that supports focal + soundscape audio directories.

    Files prefixed with __focal__/ resolve to focal_audio_dir,
    files prefixed with __soundscape__/ resolve to soundscape_audio_dir.
    """

    def __init__(
        self,
        focal_audio_dir: str,
        soundscape_audio_dir: str,
        soft_labels_df: pl.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        num_tokens: int = 625,
    ):
        self.focal_audio_dir = Path(focal_audio_dir).expanduser()
        self.soundscape_audio_dir = Path(soundscape_audio_dir).expanduser()
        self.species_list = species_list
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.num_tokens = num_tokens

        self.chunks = []
        skipped_files = 0

        grouped = soft_labels_df.group_by("file", maintain_order=True)

        for file_key, file_df in grouped:
            file_key = file_key[0] if isinstance(file_key, tuple) else file_key

            # Resolve audio path from prefix
            if file_key.startswith("__soundscape__/"):
                rel_path = file_key[len("__soundscape__/"):]
                audio_path = self.soundscape_audio_dir / rel_path
            elif file_key.startswith("__focal__/"):
                rel_path = file_key[len("__focal__/"):]
                audio_path = self.focal_audio_dir / rel_path
            else:
                audio_path = self.focal_audio_dir / file_key

            if not audio_path.exists():
                skipped_files += 1
                continue

            file_df = file_df.sort("start_time")
            start_times = file_df["start_time"].to_numpy()
            label_matrix = file_df.select(species_list).to_numpy()

            file_duration = start_times[-1] + 1.0

            chunk_start = 0.0
            while chunk_start < file_duration:
                chunk_end = chunk_start + chunk_duration
                mask = (start_times >= chunk_start) & (start_times < chunk_end)
                if mask.any():
                    chunk_labels = label_matrix[mask].mean(axis=0).astype(np.float32)
                else:
                    chunk_labels = np.zeros(self.num_classes, dtype=np.float32)

                real_duration = min(chunk_duration, file_duration - chunk_start)
                real_samples = int(real_duration * sample_rate)
                start_sample = int(chunk_start * sample_rate)

                self.chunks.append((
                    str(audio_path),
                    start_sample,
                    real_samples,
                    torch.from_numpy(chunk_labels),
                ))

                chunk_start += chunk_duration

        if skipped_files > 0:
            logger.warning(f"Skipped {skipped_files} recordings (file not found)")
        logger.info(
            f"MultiDirSoftLabelDataset: {len(self.chunks)} chunks from "
            f"{soft_labels_df['file'].n_unique()} recordings ({self.num_classes} classes)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, real_samples, label_vec = self.chunks[idx]

        waveform = load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=start_sample, num_samples=real_samples,
        )

        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        hop_samples = self.chunk_samples // self.num_tokens
        real_tokens = min(
            (real_samples + hop_samples - 1) // hop_samples,
            self.num_tokens,
        )

        return {
            "waveform": waveform,
            "clip_labels": label_vec.clone(),
            "num_tokens": self.num_tokens,
            "real_tokens": int(real_tokens),
        }
