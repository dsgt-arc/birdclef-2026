"""Data loading for experiment 107 training pipeline.

Wraps the exp 106 parquet cache to provide the three training streams:
  1. Strong labeled soundscapes (739 windows with ground truth)
  2. Weak focal recordings (~265K windows with bag-level labels)
  3. Unsupervised soundscapes (~127K windows with site metadata only)

Also provides LOSO fold splitting for cross-validation on labeled soundscapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    list_sites,
    load_focal_shards,
    load_labeled,
    load_unlabeled_shards,
)

# Default features directory on scratch (exp 106 cache).
FEATURES_DIR = "~/scratch/birdclef-2026/features/exp106"

# Artifacts directory for exp 106 (species mapping JSON).
# Resolved relative to the exp 106 package install location.
from pathlib import Path as _Path

_EXP106_DIR = _Path(__file__).resolve().parents[3] / "106-perch-bayesian"
ARTIFACTS_DIR = str(_EXP106_DIR / "artifacts")

def get_species_list(artifacts_dir: str | None = None) -> list[str]:
    """Canonical sorted list of 234 BirdCLEF 2026 species."""
    d = artifacts_dir or ARTIFACTS_DIR
    return build_species_list(d)


@dataclass
class LabeledSoundscapeData:
    """All 739 labeled soundscape windows, ready for training."""

    embeddings: np.ndarray  # (N, 1536) float32
    labels: np.ndarray  # (N, 234) int8
    sites: np.ndarray  # (N,) object array of site ID strings
    filenames: list[str]  # (N,)
    species_list: list[str]  # (234,)
    site_to_idx: dict[str, int]  # site string → contiguous int


def load_labeled_soundscape(
    features_dir: str = FEATURES_DIR,
) -> LabeledSoundscapeData:
    """Load all labeled soundscape windows with embeddings, labels, and sites."""
    df = load_labeled(features_dir)
    species_list = get_species_list()
    embeddings = embeddings_to_numpy(df)
    labels = build_binary_label_matrix(df, species_list)
    filenames = df["filename"].to_list()
    sites = np.array([extract_site(f) for f in filenames])

    # Build contiguous site index from the union of labeled + unlabeled sites.
    # Some labeled sites (e.g. S08, S23) may not appear in by_site/ parquets.
    unlabeled_sites = set(list_sites(features_dir))
    labeled_sites = set(sites.tolist())
    all_sites = sorted(unlabeled_sites | labeled_sites)
    site_to_idx = {s: i for i, s in enumerate(all_sites)}

    return LabeledSoundscapeData(
        embeddings=embeddings,
        labels=labels,
        sites=sites,
        filenames=filenames,
        species_list=species_list,
        site_to_idx=site_to_idx,
    )


@dataclass
class FocalData:
    """All focal recording windows, concatenated from shards."""

    embeddings: np.ndarray  # (N, 1536) float32
    labels: np.ndarray  # (N, 234) int8
    species_list: list[str]


def load_all_focal(features_dir: str = FEATURES_DIR) -> FocalData:
    """Load all focal recording shards into memory.

    ~265K windows × 1536 dims × 4 bytes ≈ 1.6 GB. Fits in a 6 GB job.
    """
    species_list = get_species_list()
    all_emb = []
    all_labels = []
    for shard_df in load_focal_shards(features_dir):
        all_emb.append(embeddings_to_numpy(shard_df))
        all_labels.append(build_binary_label_matrix(shard_df, species_list))
    return FocalData(
        embeddings=np.concatenate(all_emb, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        species_list=species_list,
    )


@dataclass
class UnlabeledSoundscapeData:
    """All unlabeled soundscape windows with site metadata."""

    embeddings: np.ndarray  # (N, 1536) float32
    site_indices: np.ndarray  # (N,) int64 — contiguous site index
    site_to_idx: dict[str, int]


def load_all_unlabeled(
    features_dir: str = FEATURES_DIR,
    site_to_idx: dict[str, int] | None = None,
) -> UnlabeledSoundscapeData:
    """Load all unlabeled soundscape shards with site indices.

    ~127K windows × 1536 dims × 4 bytes ≈ 0.75 GB.
    """
    if site_to_idx is None:
        all_sites = sorted(list_sites(features_dir))
        site_to_idx = {s: i for i, s in enumerate(all_sites)}

    all_emb = []
    all_site_idx = []
    for shard_df in load_unlabeled_shards(features_dir):
        all_emb.append(embeddings_to_numpy(shard_df))
        fnames = shard_df["filename"].to_list()
        sites = [extract_site(f) for f in fnames]
        all_site_idx.append(
            np.array([site_to_idx.get(s, 0) for s in sites], dtype=np.int64)
        )
    return UnlabeledSoundscapeData(
        embeddings=np.concatenate(all_emb, axis=0),
        site_indices=np.concatenate(all_site_idx, axis=0),
        site_to_idx=site_to_idx,
    )


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------


@dataclass
class LOSOFold:
    """One leave-one-site-out fold."""

    left_out_site: str
    train_mask: np.ndarray  # (N,) bool
    test_mask: np.ndarray  # (N,) bool


def loso_folds(sites: np.ndarray) -> list[LOSOFold]:
    """Generate LOSO folds from site labels.

    Only yields folds for sites that have at least one labeled window.
    Returns folds sorted by site ID for reproducibility.
    """
    unique_sites = sorted(set(sites.tolist()))
    folds = []
    for site in unique_sites:
        test_mask = sites == site
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        folds.append(
            LOSOFold(left_out_site=site, train_mask=train_mask, test_mask=test_mask)
        )
    return folds


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset):
    """Simple dataset wrapping numpy arrays of embeddings + labels."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        site_indices: np.ndarray | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels
        self.site_indices = site_indices

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "emb": torch.from_numpy(self.embeddings[idx].copy()).float(),
            "labels": torch.from_numpy(self.labels[idx].copy()).float(),
        }
        if self.site_indices is not None:
            item["site_idx"] = torch.tensor(self.site_indices[idx], dtype=torch.long)
        return item


class UnsupervisedDataset(Dataset):
    """Dataset for unlabeled soundscape windows (embeddings + site index only)."""

    def __init__(
        self,
        embeddings: np.ndarray,
        site_indices: np.ndarray,
    ) -> None:
        self.embeddings = embeddings
        self.site_indices = site_indices

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "emb": torch.from_numpy(self.embeddings[idx].copy()).float(),
            "site_idx": torch.tensor(self.site_indices[idx], dtype=torch.long),
        }


def build_focal_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Per-species balanced sampler for focal recordings.

    Weight each window by the inverse frequency of its primary (most
    specific) label, so rare species are sampled as often as common ones.
    """
    # Per-window weight = 1 / (sum of species frequency for positive labels)
    species_counts = labels.sum(axis=0).astype(np.float64)  # (234,)
    species_counts = np.maximum(species_counts, 1.0)  # avoid div by zero
    species_weights = 1.0 / species_counts  # (234,)

    # Each window's weight is the max weight among its positive labels
    # (dominated by the rarest species in the window).
    per_window = np.zeros(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        pos = np.where(labels[i] > 0)[0]
        if len(pos) > 0:
            per_window[i] = species_weights[pos].max()
        else:
            per_window[i] = 1.0  # shouldn't happen, but safe fallback

    return WeightedRandomSampler(
        weights=per_window.tolist(),
        num_samples=len(labels),
        replacement=True,
    )


def collate_triple_batch(
    strong_batch: dict[str, torch.Tensor],
    focal_batch: dict[str, torch.Tensor],
    unsup_batch: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor] | None]:
    """Assemble the three-stream batch dict expected by compute_loss()."""
    return {
        "strong": {
            "emb": strong_batch["emb"],
            "site_idx": strong_batch["site_idx"],
            "labels": strong_batch["labels"],
        },
        "weak_focal": {
            "emb": focal_batch["emb"],
            "labels": focal_batch["labels"],
        },
        "unsup": {
            "emb": unsup_batch["emb"],
            "site_idx": unsup_batch["site_idx"],
        },
    }
