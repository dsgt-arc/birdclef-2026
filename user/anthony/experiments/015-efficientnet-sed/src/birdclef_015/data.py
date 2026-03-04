"""SED soft-label dataset: loads fused parquets, returns per-frame (5, C) labels."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader, Dataset

from birdclef_009.data.audio import load_audio_chunk
from birdclef_009.data.finetune import BirdCLEFTrainDataset, build_species_list

logger = logging.getLogger(__name__)

NUM_FRAMES = 5  # 5 × 1s frames per 5s chunk


def collate_sed(batch: list[dict]) -> dict:
    """Collate function for SED training with frame labels.

    Returns:
        waveform: (B, num_samples)
        clip_labels: (B, C)
        frame_labels: (B, 5, C)
        frame_mask: (B, 5) — True for frames with real audio
        padding_mask: (B, T) — mel token padding mask
    """
    waveforms = torch.stack([item["waveform"] for item in batch])
    clip_labels = torch.stack([item["clip_labels"] for item in batch])
    frame_labels = torch.stack([item["frame_labels"] for item in batch])

    num_tokens = batch[0]["num_tokens"]
    real_tokens_list = [item["real_tokens"] for item in batch]
    padding_mask = torch.zeros(len(batch), num_tokens, dtype=torch.bool)
    for i, rt in enumerate(real_tokens_list):
        padding_mask[i, :rt] = True

    # Frame mask: which of the 5 frames have real audio
    real_frames_list = [item["real_frames"] for item in batch]
    frame_mask = torch.zeros(len(batch), NUM_FRAMES, dtype=torch.bool)
    for i, rf in enumerate(real_frames_list):
        frame_mask[i, :rf] = True

    return {
        "waveform": waveforms,
        "clip_labels": clip_labels,
        "frame_labels": frame_labels,
        "frame_mask": frame_mask,
        "padding_mask": padding_mask,
    }


class SEDSoftLabelDataset(Dataset):
    """Load audio chunks with per-frame soft labels from fused parquets.

    Soft labels are at 1s resolution. For each 5s audio chunk, returns
    frame_labels: (5, C) per-second soft labels and clip_labels: (C,) as their mean.

    Args:
        audio_dir: Root directory containing audio files (species/recording.ogg)
        soft_labels_df: polars DataFrame with columns [file, start_time, ...species...]
        species_list: Ordered list of species codes
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

        self.chunks = []
        skipped_files = 0

        grouped = soft_labels_df.group_by("file", maintain_order=True)

        for file_key, file_df in grouped:
            file_key = file_key[0] if isinstance(file_key, tuple) else file_key
            audio_path = self.audio_dir / file_key
            if not audio_path.exists():
                skipped_files += 1
                continue

            file_df = file_df.sort("start_time")
            start_times = file_df["start_time"].to_numpy()
            label_matrix = file_df.select(species_list).to_numpy()  # (T_file, C)

            file_duration = start_times[-1] + 1.0

            chunk_start = 0.0
            while chunk_start < file_duration:
                chunk_end = chunk_start + chunk_duration

                # Build per-frame labels (5 × 1s frames)
                frame_labels = []
                for t in range(NUM_FRAMES):
                    token_start = chunk_start + t
                    token_end = token_start + 1.0
                    mask = (start_times >= token_start) & (start_times < token_end)
                    if mask.any():
                        frame_labels.append(label_matrix[mask].mean(axis=0))
                    else:
                        frame_labels.append(np.zeros(self.num_classes))
                frame_labels = np.stack(frame_labels).astype(np.float32)  # (5, C)

                # How many of the 5 frames have real audio
                real_duration = min(chunk_duration, file_duration - chunk_start)
                real_samples = int(real_duration * sample_rate)
                real_frames = min(NUM_FRAMES, int(np.ceil(real_duration)))

                start_sample = int(chunk_start * sample_rate)
                self.chunks.append((
                    str(audio_path),
                    start_sample,
                    real_samples,
                    torch.from_numpy(frame_labels),
                    real_frames,
                ))

                chunk_start += chunk_duration

        if skipped_files > 0:
            logger.warning(f"Skipped {skipped_files} recordings (file not found)")
        logger.info(
            f"SEDSoftLabelDataset: {len(self.chunks)} chunks from "
            f"{soft_labels_df['file'].n_unique()} recordings ({self.num_classes} classes)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, real_samples, frame_labels, real_frames = self.chunks[idx]

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
            "clip_labels": frame_labels.mean(dim=0),   # (C,)
            "frame_labels": frame_labels,                # (5, C)
            "num_tokens": self.num_tokens,
            "real_tokens": int(real_tokens),
            "real_frames": int(real_frames),
        }


class _MultiDirSEDSoftLabelDataset(Dataset):
    """SEDSoftLabelDataset variant supporting focal + soundscape audio directories."""

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
                frame_labels = []
                for t in range(NUM_FRAMES):
                    token_start = chunk_start + t
                    token_end = token_start + 1.0
                    mask = (start_times >= token_start) & (start_times < token_end)
                    if mask.any():
                        frame_labels.append(label_matrix[mask].mean(axis=0))
                    else:
                        frame_labels.append(np.zeros(self.num_classes))
                frame_labels = np.stack(frame_labels).astype(np.float32)

                real_duration = min(chunk_duration, file_duration - chunk_start)
                real_samples = int(real_duration * sample_rate)
                real_frames = min(NUM_FRAMES, int(np.ceil(real_duration)))
                start_sample = int(chunk_start * sample_rate)

                self.chunks.append((
                    str(audio_path),
                    start_sample,
                    real_samples,
                    torch.from_numpy(frame_labels),
                    real_frames,
                ))

                chunk_start += chunk_duration

        if skipped_files > 0:
            logger.warning(f"Skipped {skipped_files} recordings (file not found)")
        logger.info(
            f"MultiDirSEDSoftLabelDataset: {len(self.chunks)} chunks from "
            f"{soft_labels_df['file'].n_unique()} recordings ({self.num_classes} classes)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, real_samples, frame_labels, real_frames = self.chunks[idx]

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
            "clip_labels": frame_labels.mean(dim=0),
            "frame_labels": frame_labels,
            "num_tokens": self.num_tokens,
            "real_tokens": int(real_tokens),
            "real_frames": int(real_frames),
        }


class SEDSoftLabelDataModule(ptl.LightningDataModule):
    """DataModule with per-frame soft labels for train, hard labels for val.

    Train: SEDSoftLabelDataset with fused parquet soft labels (5 × 1s frames)
    Val: BirdCLEFTrainDataset with hard labels from train.csv (same split as exp 012)
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

        if self.include_soundscape and not self.soundscape_audio_dir:
            raise ValueError("soundscape_audio_dir must be set when include_soundscape=True")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Val split: same stratified split as exp 012
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

            # Val dataset: hard labels (same as exp 012)
            self.val_dataset = BirdCLEFTrainDataset(
                audio_dir=self.audio_dir,
                metadata_df=val_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                num_tokens=self.num_tokens,
            )

            # Train dataset: per-frame soft labels
            focal_path = self.soft_labels_dir / "focal" / "soft_labels.parquet"
            logger.info(f"Loading focal soft labels from {focal_path}")
            focal_df = pl.read_parquet(focal_path)
            logger.info(f"Focal soft labels: {focal_df.shape}")

            val_filenames = set(val_df["filename"].tolist())
            focal_df = focal_df.filter(~pl.col("file").is_in(val_filenames))
            logger.info(f"Focal after excluding val: {focal_df.shape}")

            if self.include_soundscape:
                sc_path = self.soft_labels_dir / "soundscape" / "soft_labels.parquet"
                logger.info(f"Loading soundscape soft labels from {sc_path}")
                sc_df = pl.read_parquet(sc_path)
                logger.info(f"Soundscape soft labels: {sc_df.shape}")

                sc_df = sc_df.with_columns(
                    (pl.lit("__soundscape__/").cast(pl.Utf8) + pl.col("file")).alias("file")
                )
                focal_df = focal_df.with_columns(
                    (pl.lit("__focal__/").cast(pl.Utf8) + pl.col("file")).alias("file")
                )
                combined_df = pl.concat([focal_df, sc_df])
                logger.info(f"Combined train soft labels: {combined_df.shape}")

                self.train_dataset = _MultiDirSEDSoftLabelDataset(
                    focal_audio_dir=self.audio_dir,
                    soundscape_audio_dir=self.soundscape_audio_dir,
                    soft_labels_df=combined_df,
                    species_list=self.species_list,
                    sample_rate=self.sample_rate,
                    chunk_duration=self.chunk_duration,
                    num_tokens=self.num_tokens,
                )
            else:
                self.train_dataset = SEDSoftLabelDataset(
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
            collate_fn=collate_sed,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        from birdclef_009.data.audio import collate_finetune

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
