"""BirdCLEF dataset and Lightning DataModule.

Loads local BirdCLEF audio (train_audio/ with species subdirectories), splits each
recording into non-overlapping 5s chunks, resamples to target sample rate, and
returns (waveform, labels) tuples for pretrained encoder fine-tuning.

Compatible with BirdCLEF 2025 and 2026 data formats (same CSV columns and
audio directory structure).
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


def build_species_list(metadata_csv: str | Path) -> list[str]:
    """Extract sorted list of unique species from metadata CSV."""
    df = pd.read_csv(metadata_csv)
    species = sorted(df["primary_label"].unique().tolist())
    logger.info(f"Found {len(species)} species in {metadata_csv}")
    return species


def _load_audio_chunk(audio_path, target_sr, start_sample=0, num_samples=None):
    """Load a chunk of audio and resample to target sample rate.

    Uses sf.read(start, frames) for partial reads to avoid loading entire files.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        start_sample: Start position in target sample rate
        num_samples: Number of samples to read in target sample rate (None = to end)
    """
    info = sf.info(str(audio_path))
    source_sr = info.samplerate

    # Convert target-domain positions to source-domain
    src_start = int(start_sample * source_sr / target_sr)
    if num_samples is not None:
        # Read a bit extra to avoid off-by-one after resampling
        src_frames = int(num_samples * source_sr / target_sr) + 2
    else:
        src_frames = -1  # read to end

    audio, _ = sf.read(str(audio_path), dtype="float32", start=src_start, frames=src_frames)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=source_sr, new_freq=target_sr
        )

    # Trim to exact requested length (resampling can produce ±1 sample)
    if num_samples is not None and audio_tensor.shape[0] > num_samples:
        audio_tensor = audio_tensor[:num_samples]

    return audio_tensor


class BirdCLEFDataset(Dataset):
    """BirdCLEF 2025 dataset: non-overlapping 5s chunks at 16kHz.

    Each recording is split into non-overlapping chunks. Labels are multi-hot
    vectors from primary_label + secondary_labels.

    Returns:
        (waveform, labels) — waveform is (chunk_samples,) at 16kHz,
        labels is (num_classes,) multi-hot float tensor.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        metadata_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        transform=None,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.transform = transform

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

            # Get file duration in target sample rate
            try:
                info = sf.info(str(audio_path))
                total_samples = int(info.frames * sample_rate / info.samplerate)
            except Exception:
                skipped += 1
                continue

            # Split into non-overlapping chunks
            pos = 0
            while pos < total_samples:
                real_samples = min(self.chunk_samples, total_samples - pos)
                self.chunks.append((str(audio_path), pos, real_samples, label_vec))
                pos += self.chunk_samples

        if skipped > 0:
            logger.warning(f"Skipped {skipped} recordings (file not found or unreadable)")
        logger.info(
            f"BirdCLEFDataset: {len(self.chunks)} chunks from "
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
            # Parse secondary labels (format: "['sp1', 'sp2']")
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

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=start_sample,
            num_samples=real_samples,
        )

        # Pad to chunk_samples if needed
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        # Apply optional transform (e.g., fbank conversion for EAT)
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label_vec.clone()


class BirdCLEFDataModule(pl.LightningDataModule):
    """Lightning DataModule for BirdCLEF 2025 labeled training data.

    Reads train.csv, does stratified train/val/test split by primary_label,
    and creates BirdCLEFDataset for each split.

    Split strategy (80/10/10 by default):
        - Train: model training
        - Val: early stopping, checkpoint selection, HP tuning
        - Test: held-out final evaluation (never used during training)
    """

    def __init__(
        self,
        audio_dir: str | Path,
        metadata_csv: str | Path,
        species_list: list[str] | None = None,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        batch_size: int = 32,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["species_list", "transform"])
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.transform = transform

        # Build or use provided species list
        if species_list is not None:
            self.species_list = species_list
        else:
            self.species_list = build_species_list(metadata_csv)

        self.num_classes = len(self.species_list)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _split_dataframe(self, df: pd.DataFrame):
        """Stratified three-way split by primary_label.

        Guarantees every species has at least 1 sample in each split.
        For species with fewer than 3 samples, duplicates are placed into
        the deficient splits so no class is absent from any split.
        """
        rng = np.random.RandomState(self.seed)
        train_indices = []
        val_indices = []
        test_indices = []
        duplicated_species = []

        for species, group in df.groupby("primary_label"):
            indices = group.index.tolist()
            rng.shuffle(indices)
            n = len(indices)

            if n >= 3:
                # Normal case: allocate at least 1 to test and val, rest to train
                n_test = max(1, int(n * self.test_fraction))
                n_val = max(1, int(n * self.val_fraction))
                # Ensure train gets at least 1
                if n_test + n_val >= n:
                    n_test = 1
                    n_val = 1
                test_indices.extend(indices[:n_test])
                val_indices.extend(indices[n_test : n_test + n_val])
                train_indices.extend(indices[n_test + n_val :])
            elif n == 2:
                # 2 samples: one to train, one to val, duplicate train sample to test
                train_indices.append(indices[0])
                val_indices.append(indices[1])
                test_indices.append(indices[0])  # duplicate
                duplicated_species.append(species)
            else:
                # 1 sample: put in all three splits (rare edge case)
                train_indices.append(indices[0])
                val_indices.append(indices[0])
                test_indices.append(indices[0])
                duplicated_species.append(species)

        if duplicated_species:
            logger.warning(
                f"{len(duplicated_species)} species had <3 samples and required "
                f"duplication across splits: {duplicated_species[:5]}"
                + ("..." if len(duplicated_species) > 5 else "")
            )

        train_df = df.loc[train_indices].reset_index(drop=True)
        val_df = df.loc[val_indices].reset_index(drop=True)
        test_df = df.loc[test_indices].reset_index(drop=True)

        return train_df, val_df, test_df

    def setup(self, stage=None):
        # Only build datasets once (they're deterministic given the seed)
        if (
            self.train_dataset is not None
            and self.val_dataset is not None
            and self.test_dataset is not None
        ):
            return

        df = pd.read_csv(self.metadata_csv)
        train_df, val_df, test_df = self._split_dataframe(df)

        logger.info(
            f"Split: {len(train_df)} train, {len(val_df)} val, " f"{len(test_df)} test recordings"
        )

        self.train_dataset = BirdCLEFDataset(
            audio_dir=self.audio_dir,
            metadata_df=train_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )
        self.val_dataset = BirdCLEFDataset(
            audio_dir=self.audio_dir,
            metadata_df=val_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
        )
        self.test_dataset = BirdCLEFDataset(
            audio_dir=self.audio_dir,
            metadata_df=test_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            transform=self.transform,
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
