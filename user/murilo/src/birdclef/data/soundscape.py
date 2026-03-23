"""Soundscape dataset and utilities for BirdCLEF 2026 labeled soundscapes.

Loads 5-second segments from expert-annotated soundscape files, with multi-label
annotations from train_soundscapes_labels.csv. Also provides FocalDataset234 wrapper
to remap 206-dim focal labels to 234-dim for mixed training.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import _load_audio_chunk

logger = logging.getLogger(__name__)


def build_full_species_list(taxonomy_csv: str | Path) -> list[str]:
    """Build sorted list of all 234 submission species from taxonomy.csv."""
    df = pd.read_csv(taxonomy_csv)
    species = sorted(df["primary_label"].unique().tolist())
    logger.info(f"Found {len(species)} species in {taxonomy_csv}")
    return species


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS timestamp to seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def load_soundscape_labels(labels_csv: str | Path) -> pd.DataFrame:
    """Load and deduplicate soundscape labels.

    The CSV contains duplicate rows (each segment appears 2x with identical labels).
    Deduplicates by (filename, start, end), keeping the first occurrence.

    Returns:
        DataFrame with columns: filename, start, end, primary_label (deduplicated)
    """
    df = pd.read_csv(labels_csv)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["filename", "start", "end"], keep="first")
    n_dedup = len(df)
    logger.info(
        f"Soundscape labels: {n_raw} rows → {n_dedup} unique segments "
        f"from {df['filename'].nunique()} files"
    )
    return df


class SoundscapeDataset(Dataset):
    """Dataset for BirdCLEF 2026 labeled soundscape segments.

    Each sample is a 5-second segment from a 1-minute soundscape file, with
    multi-label annotations (semicolon-separated species IDs).

    Returns:
        (waveform_or_features, labels) — labels is (234,) multi-hot float tensor.
    """

    def __init__(
        self,
        soundscape_dir: str | Path,
        labels_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        transform=None,
    ):
        self.soundscape_dir = Path(soundscape_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.transform = transform

        # Build chunk index from labels
        self.chunks = []
        skipped = 0
        unmapped_species = set()

        for _, row in labels_df.iterrows():
            audio_path = self.soundscape_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            start_sec = _parse_timestamp(row["start"])
            start_sample = int(start_sec * sample_rate)
            num_samples = self.chunk_samples

            # Build multi-hot label vector from semicolon-separated species
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for species_id in str(row["primary_label"]).split(";"):
                species_id = species_id.strip()
                if species_id in self.species_to_idx:
                    label_vec[self.species_to_idx[species_id]] = 1.0
                else:
                    unmapped_species.add(species_id)

            self.chunks.append((str(audio_path), start_sample, num_samples, label_vec))

        if skipped > 0:
            logger.warning(f"Skipped {skipped} segments (audio file not found)")
        if unmapped_species:
            logger.warning(
                f"{len(unmapped_species)} species in labels not in species list: "
                f"{sorted(unmapped_species)[:10]}"
            )
        logger.info(
            f"SoundscapeDataset: {len(self.chunks)} segments from "
            f"{labels_df['filename'].nunique()} files ({self.num_classes} classes)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, num_samples, label_vec = self.chunks[idx]

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=start_sample,
            num_samples=num_samples,
        )

        # Pad to exact chunk_samples if needed
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        # Apply optional transform (e.g., fbank conversion for EAT)
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label_vec.clone()


class FocalDataset234(Dataset):
    """Wrapper that remaps 206-dim focal labels to 234-dim.

    Wraps an existing BirdCLEFDataset (206 classes) and remaps its multi-hot
    labels to the full 234-species list. The 28 new species always get 0.0
    (no focal training data exists for them).
    """

    def __init__(
        self,
        focal_dataset: Dataset,
        species_206: list[str],
        species_234: list[str],
    ):
        self.focal_dataset = focal_dataset
        self.num_classes = len(species_234)

        # Build remapping: for each old index, what's the new index?
        idx_234 = {s: i for i, s in enumerate(species_234)}
        self.remap_indices = []
        for s in species_206:
            if s in idx_234:
                self.remap_indices.append(idx_234[s])
            else:
                logger.warning(f"Focal species '{s}' not found in 234-species list")
                self.remap_indices.append(-1)

        n_mapped = sum(1 for i in self.remap_indices if i >= 0)
        logger.info(
            f"FocalDataset234: remapping {len(species_206)} → {len(species_234)} "
            f"species ({n_mapped} mapped)"
        )

    def __len__(self):
        return len(self.focal_dataset)

    def __getitem__(self, idx):
        waveform, label_206 = self.focal_dataset[idx]
        label_234 = torch.zeros(self.num_classes, dtype=torch.float32)

        for old_idx, new_idx in enumerate(self.remap_indices):
            if new_idx >= 0:
                label_234[new_idx] = label_206[old_idx]

        return waveform, label_234
