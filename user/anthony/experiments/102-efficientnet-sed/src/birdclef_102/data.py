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

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

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

    If mel_transform is provided, computes mel spectrogram on CPU in the
    DataLoader workers (frees GPU VRAM for larger batches). Otherwise
    returns raw waveforms for GPU-side mel computation.

    Each item returns:
        waveform: (num_samples,) audio tensor  [if no mel_transform]
        mel: (1, n_mels, T) mel spectrogram    [if mel_transform provided]
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
        mel_transform: Callable | None = None,
        target_frames: int | None = None,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        soundscape_dir: str | None = None,
        soundscape_labels_csv: str | None = None,
        nocall_parquet: str | None = None,
        teacher_predictions: str | None = None,
        teacher_threshold: float = 0.0,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.mel_transform = mel_transform
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.teacher_predictions = teacher_predictions
        self.teacher_threshold = teacher_threshold

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

        # Append soundscape chunks (labeled windows + no-call windows)
        if soundscape_dir and soundscape_labels_csv:
            self._append_soundscape_chunks(soundscape_dir, soundscape_labels_csv)
        if soundscape_dir and nocall_parquet and Path(nocall_parquet).exists():
            self._append_nocall_chunks(soundscape_dir, nocall_parquet)

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

        # Optional: join with teacher predictions for filtering
        teacher_lookup = None
        if self.teacher_predictions and Path(self.teacher_predictions).exists() and self.teacher_threshold > 0:
            teacher = pd.read_parquet(self.teacher_predictions)
            teacher_lookup = {}
            for _, row in teacher.iterrows():
                teacher_lookup[(row["filename"], row["chunk_start"])] = float(row["teacher_prob"])
            logger.info(f"Loaded teacher predictions: {len(teacher_lookup)} chunks, threshold={self.teacher_threshold}")

        # Build chunks using relpath for correct label lookup
        self.chunks = []
        filtered_count = 0
        for basename, group in activity.groupby("filename"):
            relpath = self.file_to_relpath.get(basename)
            if relpath is None or relpath not in self.file_to_labels:
                continue

            audio_path = str(self.audio_dir / relpath)
            label_vec = self.file_to_labels[relpath]

            for _, row in group.iterrows():
                chunk_start = float(row["chunk_start"])

                # Filter by teacher prediction
                if teacher_lookup is not None:
                    tp = teacher_lookup.get((basename, chunk_start))
                    if tp is not None and tp < self.teacher_threshold:
                        filtered_count += 1
                        continue

                start_sample = int(chunk_start * self.sample_rate)
                activity_score = float(row.get("activity_score", 0.5))
                self.chunks.append((
                    audio_path,
                    start_sample,
                    self.chunk_samples,
                    label_vec,
                    activity_score,
                ))

        if filtered_count > 0:
            logger.info(f"Teacher filtering: dropped {filtered_count} chunks (threshold={self.teacher_threshold})")

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

    def _append_soundscape_chunks(self, soundscape_dir, labels_csv):
        """Append labeled soundscape windows (multi-label, semicolon-separated IDs)."""
        sdir = Path(soundscape_dir)
        labels = pd.read_csv(labels_csv)

        def time_to_sec(t):
            parts = t.split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        # Deduplicate — labels CSV has duplicate rows per (filename, start)
        labels = labels.drop_duplicates(subset=["filename", "start"])

        count = 0
        for _, row in labels.iterrows():
            audio_path = sdir / row["filename"]
            if not audio_path.exists():
                continue

            start_sec = time_to_sec(row["start"])
            start_sample = int(start_sec * self.sample_rate)

            # Build multi-label vector from semicolon-separated species IDs
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for sid in str(row["primary_label"]).split(";"):
                sid = sid.strip()
                if sid in self.species_to_idx:
                    label_vec[self.species_to_idx[sid]] = 1.0

            # Skip windows where no focal species are present
            if label_vec.sum() == 0:
                continue

            self.chunks.append((
                str(audio_path),
                start_sample,
                self.chunk_samples,
                label_vec,
                1.0,  # high activity — annotator confirmed species present
            ))
            count += 1

        # Count how many chunks actually have positive labels
        pos_count = sum(1 for c in self.chunks[-count:] if c[3].sum() > 0) if count > 0 else 0
        logger.info(f"Appended {count} labeled soundscape windows ({pos_count} with focal species)")

    def _append_nocall_chunks(self, soundscape_dir, nocall_parquet):
        """Append no-call windows (all-zeros labels) from labeled soundscapes."""
        sdir = Path(soundscape_dir)
        nocall = pd.read_parquet(nocall_parquet)
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        count = 0
        for _, row in nocall.iterrows():
            audio_path = sdir / row["filename"]
            if not audio_path.exists():
                continue

            start_sample = int(row["chunk_start"] * self.sample_rate)
            self.chunks.append((
                str(audio_path),
                start_sample,
                self.chunk_samples,
                label_vec.clone(),
                0.0,  # no activity
            ))
            count += 1

        logger.info(f"Appended {count} no-call soundscape windows")

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

        if self.mel_transform is not None:
            # Compute mel on CPU in dataloader worker
            mel = self.mel_transform(waveform.unsqueeze(0))  # (1, n_mels, T)
            mel = torch.log(mel.clamp(min=1e-10))
            mel = (mel - self.norm_mean) / (self.norm_std * 2)
            # Pad/trim to target frames
            if self.target_frames is not None:
                T = mel.shape[2]
                if T > self.target_frames:
                    mel = mel[:, :, :self.target_frames]
                elif T < self.target_frames:
                    mel = torch.nn.functional.pad(mel, (0, self.target_frames - T))
            return {
                "mel": mel.squeeze(0),  # (n_mels, T) — will be stacked to (B, n_mels, T)
                "clip_labels": label_vec.clone(),
                "activity_score": activity,
            }

        return {
            "waveform": waveform,
            "clip_labels": label_vec.clone(),
            "activity_score": activity,
        }


def collate_fn(batch):
    """Collate batch with activity scores. Handles both waveform and mel modes."""
    clip_labels = torch.stack([item["clip_labels"] for item in batch])
    activity_scores = torch.tensor(
        [item["activity_score"] for item in batch], dtype=torch.float32
    )

    result = {
        "clip_labels": clip_labels,
        "activity_scores": activity_scores,
    }

    if "mel" in batch[0]:
        # Pre-computed mel: (B, n_mels, T) → add channel dim → (B, 1, n_mels, T)
        result["mel"] = torch.stack([item["mel"] for item in batch]).unsqueeze(1)
    else:
        result["waveform"] = torch.stack([item["waveform"] for item in batch])

    return result


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
        mel_transform: Callable | None = None,
        target_frames: int | None = None,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        soundscape_dir: str | None = None,
        soundscape_labels_csv: str | None = None,
        nocall_parquet: str | None = None,
        val_on_soundscape: bool = False,
        teacher_predictions: str | None = None,
        teacher_threshold: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["species_list", "mel_transform"])
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.activity_parquet = activity_parquet
        self.mel_transform = mel_transform
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.soundscape_dir = soundscape_dir
        self.soundscape_labels_csv = soundscape_labels_csv
        self.nocall_parquet = nocall_parquet
        self.val_on_soundscape = val_on_soundscape
        self.teacher_predictions = teacher_predictions
        self.teacher_threshold = teacher_threshold

        if species_list is not None:
            self.species_list = species_list
        else:
            self.species_list = build_species_list(metadata_csv)

        self.num_classes = len(self.species_list)

    def _split_soundscape_files(self, seed):
        """Split soundscape labels CSV by file into train/val (50/50).

        Returns (train_csv_path, val_csv_path, train_nocall_path, val_nocall_path).
        Writes temporary split CSVs to the same directory as the original.
        """
        import tempfile

        labels = pd.read_csv(self.soundscape_labels_csv)
        files = sorted(labels["filename"].unique())
        rng = np.random.RandomState(seed)
        rng.shuffle(files)

        mid = len(files) // 2
        train_files = set(files[:mid])
        val_files = set(files[mid:])

        logger.info(f"Soundscape split: {len(train_files)} train files, {len(val_files)} val files")

        # Split labels
        train_labels = labels[labels["filename"].isin(train_files)]
        val_labels = labels[labels["filename"].isin(val_files)]

        # Write split CSVs to temp files
        train_csv = tempfile.NamedTemporaryFile(suffix="_sc_train.csv", delete=False, mode="w")
        val_csv = tempfile.NamedTemporaryFile(suffix="_sc_val.csv", delete=False, mode="w")
        train_labels.to_csv(train_csv.name, index=False)
        val_labels.to_csv(val_csv.name, index=False)

        # Split nocall parquet if it exists
        train_nocall = None
        val_nocall = None
        if self.nocall_parquet and Path(self.nocall_parquet).exists():
            nocall = pd.read_parquet(self.nocall_parquet)
            nc_train = nocall[nocall["filename"].isin(train_files)]
            nc_val = nocall[nocall["filename"].isin(val_files)]

            train_pq = tempfile.NamedTemporaryFile(suffix="_nc_train.parquet", delete=False)
            val_pq = tempfile.NamedTemporaryFile(suffix="_nc_val.parquet", delete=False)
            nc_train.to_parquet(train_pq.name, index=False)
            nc_val.to_parquet(val_pq.name, index=False)
            train_nocall = train_pq.name
            val_nocall = val_pq.name

            logger.info(f"Nocall split: {len(nc_train)} train, {len(nc_val)} val")

        return train_csv.name, val_csv.name, train_nocall, val_nocall

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

            dataset_kwargs = dict(
                audio_dir=self.audio_dir,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                activity_parquet=self.activity_parquet,
                mel_transform=self.mel_transform,
                target_frames=self.target_frames,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                teacher_predictions=self.teacher_predictions,
                teacher_threshold=self.teacher_threshold,
            )
            # Split soundscape files for train/val if using soundscape validation
            sc_labels_train = self.soundscape_labels_csv
            sc_labels_val = None
            nocall_train = self.nocall_parquet
            nocall_val = None

            if self.val_on_soundscape and self.soundscape_labels_csv:
                sc_labels_train, sc_labels_val, nocall_train, nocall_val = \
                    self._split_soundscape_files(self.seed)

            # Soundscape data goes into training
            self.train_dataset = BirdCLEF102Dataset(
                metadata_df=train_df,
                soundscape_dir=self.soundscape_dir,
                soundscape_labels_csv=sc_labels_train,
                nocall_parquet=nocall_train,
                **dataset_kwargs,
            )

            if self.val_on_soundscape and sc_labels_val is not None:
                # Validate on held-out soundscape files
                empty_df = train_df.iloc[:0]
                self.val_dataset = BirdCLEF102Dataset(
                    metadata_df=empty_df,
                    soundscape_dir=self.soundscape_dir,
                    soundscape_labels_csv=sc_labels_val,
                    nocall_parquet=nocall_val,
                    **dataset_kwargs,
                )
                logger.info(f"Validation on soundscape: {len(self.val_dataset)} windows")
            else:
                self.val_dataset = BirdCLEF102Dataset(
                    metadata_df=val_df, **dataset_kwargs,
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
