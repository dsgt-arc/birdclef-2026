"""BirdCLEF labeled training data: focal recordings with species labels."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import collate_finetune, load_audio_chunk

logger = logging.getLogger(__name__)


def build_species_list(metadata_csv: str) -> list[str]:
    """Extract sorted list of unique species from metadata CSV."""
    df = pd.read_csv(metadata_csv)
    species = sorted(df["primary_label"].unique().tolist())
    logger.info(f"Found {len(species)} species in {metadata_csv}")
    return species


class BirdCLEFTrainDataset(Dataset):
    """Load labeled BirdCLEF focal recordings, multi-label classification.

    Each recording is split into 10s non-overlapping chunks.
    Labels are derived from primary_label (and optionally secondary_labels).
    """

    def __init__(
        self,
        audio_dir: str,
        metadata_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        chunk_duration: float = 10.0,
        num_tokens: int = 1000,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.num_tokens = num_tokens
        self.hop_s = chunk_duration / num_tokens

        # Build chunk index from metadata
        self.chunks = []
        skipped = 0
        for _, row in metadata_df.iterrows():
            audio_path = self.audio_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            # Build label vector
            label_vec = self._build_label_vec(row)

            # Get file duration
            try:
                import soundfile as sf
                info = sf.info(str(audio_path))
                total_samples = int(info.frames * sample_rate / info.samplerate)
            except Exception:
                skipped += 1
                continue

            # Split into chunks
            pos = 0
            while pos < total_samples:
                real_samples = min(self.chunk_samples, total_samples - pos)
                self.chunks.append((str(audio_path), pos, real_samples, label_vec))
                pos += self.chunk_samples

        if skipped > 0:
            logger.warning(f"Skipped {skipped} recordings (file not found or unreadable)")
        logger.info(
            f"BirdCLEFTrainDataset: {len(self.chunks)} chunks from "
            f"{len(metadata_df)} recordings ({self.num_classes} classes)"
        )

    def _build_label_vec(self, row):
        """Build multi-hot label vector from metadata row."""
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        primary = row.get("primary_label", "")
        if primary in self.species_to_idx:
            label_vec[self.species_to_idx[primary]] = 1.0

        # Include secondary labels if available
        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary.strip() and secondary != "[]":
            # Parse secondary labels (format varies: "['sp1', 'sp2']" or "sp1;sp2")
            secondary = secondary.strip("[]' ")
            for s in secondary.replace("'", "").replace('"', "").split(","):
                s = s.strip()
                if s in self.species_to_idx:
                    label_vec[self.species_to_idx[s]] = 1.0

        return label_vec

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


class BirdCLEFDataModule(pl.LightningDataModule):
    """Lightning DataModule for BirdCLEF labeled training data."""

    def __init__(
        self,
        audio_dir: str,
        metadata_csv: str,
        species_list: list[str] | None = None,
        sample_rate: int = 32000,
        chunk_duration: float = 10.0,
        num_tokens: int = 1000,
        batch_size: int = 32,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["species_list"])
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

        # Build or use provided species list
        if species_list is not None:
            self.species_list = species_list
        else:
            self.species_list = build_species_list(metadata_csv)

        self.num_classes = len(self.species_list)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            df = pd.read_csv(self.metadata_csv)

            # Stratified split by primary_label
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

            logger.info(f"Train: {len(train_df)} recordings, Val: {len(val_df)} recordings")

            self.train_dataset = BirdCLEFTrainDataset(
                audio_dir=self.audio_dir,
                metadata_df=train_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                num_tokens=self.num_tokens,
            )
            self.val_dataset = BirdCLEFTrainDataset(
                audio_dir=self.audio_dir,
                metadata_df=val_df,
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
