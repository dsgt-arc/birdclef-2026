"""Data module for exp 102: loads from pre-computed activity index.

Instead of scanning 35K audio files at startup (slow), loads the chunk index
from focal_activity.parquet (instant). Activity scores available for sample
weighting. Audio chunks loaded on-demand via soundfile partial reads.

Usage:
    dm = BirdCLEF102DataModule(
        audio_dir="/path/to/train_audio",
        metadata_csv="/path/to/train.csv",
        activity_parquet="artifacts/focal_activity.parquet",  # optional
        ...
    )
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def build_species_list(metadata_csv: str) -> list[str]:
    """Extract sorted list of unique species from metadata CSV."""
    df = pd.read_csv(metadata_csv)
    species = sorted(df["primary_label"].astype(str).unique().tolist())
    logger.info(f"Found {len(species)} species in {metadata_csv}")
    return species


def load_audio_chunk(audio_path, target_sr, start_sample=0, num_samples=None):
    """Load a chunk of audio and resample to target sample rate."""
    info = sf.info(audio_path)
    source_sr = info.samplerate

    src_start = int(start_sample * source_sr / target_sr)
    if num_samples is not None:
        src_frames = int(num_samples * source_sr / target_sr) + 2
    else:
        src_frames = -1

    audio, _ = sf.read(audio_path, dtype="float32", start=src_start, frames=src_frames)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=source_sr, new_freq=target_sr
        )
    if num_samples is not None and audio_tensor.shape[0] > num_samples:
        audio_tensor = audio_tensor[:num_samples]

    return audio_tensor


class BirdCLEF102Dataset(Dataset):
    """Dataset that loads from a pre-computed chunk index.

    If activity_parquet is provided, uses it for instant chunk enumeration
    (no file scanning). Otherwise falls back to scanning audio files.

    Each item returns:
        waveform: (num_samples,) audio tensor
        clip_labels: (num_classes,) multi-hot label vector
        activity_score: float, confidence that chunk contains a call
    """

    def __init__(
        self,
        audio_dir: str,
        metadata_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        activity_parquet: str | None = None,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)

        # Build species label lookup from metadata
        # Key by relative path (e.g., "species/XC12345.ogg") to avoid basename collisions
        self.file_to_labels = {}
        self.file_to_relpath = {}
        for _, row in metadata_df.iterrows():
            label_vec = self._build_label_vec(row)
            relpath = row["filename"]  # e.g., "species_id/recording.ogg"
            basename = Path(relpath).name
            self.file_to_labels[relpath] = label_vec
            self.file_to_relpath[basename] = relpath

        # Build chunk index
        if activity_parquet and Path(activity_parquet).exists():
            self._build_from_parquet(activity_parquet, metadata_df)
        else:
            self._build_from_scan(metadata_df)

    def _build_label_vec(self, row):
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        primary = str(row.get("primary_label", ""))
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

    def _build_from_parquet(self, parquet_path, metadata_df):
        """Build chunk index from pre-computed activity scores. Fast."""
        activity = pd.read_parquet(parquet_path)
        logger.info(f"Loading chunk index from {parquet_path}: {len(activity)} chunks")

        # Filter to only recordings in this split (by basename)
        valid_basenames = set(self.file_to_relpath.keys())
        activity = activity[activity["filename"].isin(valid_basenames)]

        # Build chunks using relpath for correct label lookup
        self.chunks = []
        for basename, group in activity.groupby("filename"):
            relpath = self.file_to_relpath.get(basename)
            if relpath is None or relpath not in self.file_to_labels:
                continue

            audio_path = str(self.audio_dir / relpath)
            label_vec = self.file_to_labels[relpath]

            for _, row in group.iterrows():
                start_sample = int(row["chunk_start"] * self.sample_rate)
                activity_score = float(row.get("activity_score", 0.5))
                self.chunks.append((
                    audio_path,
                    start_sample,
                    self.chunk_samples,
                    label_vec,
                    activity_score,
                ))

        logger.info(f"BirdCLEF102Dataset (parquet): {len(self.chunks)} chunks")

    def _build_from_scan(self, metadata_df):
        """Fallback: scan audio files for durations. Slow."""
        logger.info("No activity parquet — scanning audio files for durations...")
        self.chunks = []
        skipped = 0
        for _, row in metadata_df.iterrows():
            audio_path = self.audio_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            label_vec = self.file_to_labels.get(row["filename"])
            if label_vec is None:
                skipped += 1
                continue

            try:
                info = sf.info(str(audio_path))
                total_samples = int(info.frames * self.sample_rate / info.samplerate)
            except Exception:
                skipped += 1
                continue

            pos = 0
            while pos < total_samples:
                real_samples = min(self.chunk_samples, total_samples - pos)
                self.chunks.append((
                    str(audio_path),
                    pos,
                    real_samples,
                    label_vec,
                    0.5,  # default activity score
                ))
                pos += self.chunk_samples

        if skipped > 0:
            logger.warning(f"Skipped {skipped} recordings")
        logger.info(f"BirdCLEF102Dataset (scan): {len(self.chunks)} chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, num_samples, label_vec, activity = self.chunks[idx]

        waveform = load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=start_sample, num_samples=num_samples,
        )

        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        return {
            "waveform": waveform,
            "clip_labels": label_vec.clone(),
            "activity_score": activity,
        }


def collate_fn(batch):
    """Collate batch with activity scores."""
    waveforms = torch.stack([item["waveform"] for item in batch])
    clip_labels = torch.stack([item["clip_labels"] for item in batch])
    activity_scores = torch.tensor(
        [item["activity_score"] for item in batch], dtype=torch.float32
    )

    # Padding mask: all true for fixed 5s chunks
    num_tokens = waveforms.shape[1]  # not used by EfficientNet but kept for compat
    padding_mask = torch.ones(len(batch), 625, dtype=torch.bool)

    return {
        "waveform": waveforms,
        "clip_labels": clip_labels,
        "padding_mask": padding_mask,
        "activity_scores": activity_scores,
    }


class BirdCLEF102DataModule(pl.LightningDataModule):
    """Data module with optional activity scores from pre-computed parquet."""

    def __init__(
        self,
        audio_dir: str,
        metadata_csv: str,
        species_list: list[str] | None = None,
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        batch_size: int = 32,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        seed: int = 42,
        activity_parquet: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["species_list"])
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.activity_parquet = activity_parquet

        if species_list is not None:
            self.species_list = species_list
        else:
            self.species_list = build_species_list(metadata_csv)

        self.num_classes = len(self.species_list)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            df = pd.read_csv(self.metadata_csv)
            df["primary_label"] = df["primary_label"].astype(str)

            # Stratified split by primary_label
            rng = np.random.RandomState(self.seed)
            val_indices = []
            train_indices = []
            for _species, group in df.groupby("primary_label"):
                indices = group.index.tolist()
                rng.shuffle(indices)
                n_val = max(1, int(len(indices) * self.val_fraction))
                val_indices.extend(indices[:n_val])
                train_indices.extend(indices[n_val:])

            train_df = df.loc[train_indices].reset_index(drop=True)
            val_df = df.loc[val_indices].reset_index(drop=True)

            logger.info(f"Train: {len(train_df)} recordings, Val: {len(val_df)} recordings")

            self.train_dataset = BirdCLEF102Dataset(
                audio_dir=self.audio_dir,
                metadata_df=train_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                activity_parquet=self.activity_parquet,
            )
            self.val_dataset = BirdCLEF102Dataset(
                audio_dir=self.audio_dir,
                metadata_df=val_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                activity_parquet=self.activity_parquet,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
