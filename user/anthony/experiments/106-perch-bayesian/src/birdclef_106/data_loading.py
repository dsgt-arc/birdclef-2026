"""Shared data loading utilities for exp 106 analysis tasks.

All analysis tasks read from the same extracted parquet files on scratch.
This module centralizes loading logic so tasks stay concise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

FEATURES_DIR = "~/scratch/birdclef-2026/features/exp106"

# Filename pattern: BC2026_Train_{id}_{site}_{date}_{time}.ogg
_SITE_RE = re.compile(r"_(S\d+)_")


def load_labeled(features_dir: str = FEATURES_DIR) -> pl.DataFrame:
    """Read soundscape_labeled.parquet (739 rows with ground truth)."""
    path = Path(features_dir).expanduser() / "soundscape_labeled.parquet"
    return pl.read_parquet(path)


def load_unlabeled_shards(features_dir: str = FEATURES_DIR) -> Iterator[pl.DataFrame]:
    """Yield soundscape_unlabeled/shard-*.parquet one at a time.

    16 shards, ~8K rows each. Iterator avoids loading all 127K rows at once.
    """
    shard_dir = Path(features_dir).expanduser() / "soundscape_unlabeled"
    for path in sorted(shard_dir.glob("shard-*.parquet")):
        yield pl.read_parquet(path)


def load_focal_shards(features_dir: str = FEATURES_DIR) -> Iterator[pl.DataFrame]:
    """Yield focal/shard-*.parquet one at a time.

    16 shards, ~16K rows each. Iterator avoids loading all 266K rows at once.
    """
    shard_dir = Path(features_dir).expanduser() / "focal"
    for path in sorted(shard_dir.glob("shard-*.parquet")):
        yield pl.read_parquet(path)


def load_species_mapping(artifacts_dir: str = "artifacts") -> dict:
    """Read perch_species_mapping.json.

    Returns dict with keys:
        direct_map: {birdclef_label: perch_logit_index} (203 entries)
        genus_proxy_map: {birdclef_label: [perch_indices]} (6 entries)
        unmapped: [birdclef_labels] (25 entries)
        birdclef_species: ordered list of all 234 labels
        taxa_classes: {birdclef_label: class_name}
    """
    path = Path(artifacts_dir) / "perch_species_mapping.json"
    with open(path) as f:
        return json.load(f)


def extract_site(filename: str) -> str:
    """Extract site ID (e.g. 'S08') from soundscape filename.

    Pattern: BC2026_Train_{id}_{site}_{date}_{time}.ogg
    """
    m = _SITE_RE.search(filename)
    return m.group(1) if m else "unknown"


def build_species_list(artifacts_dir: str = "artifacts") -> list[str]:
    """Sorted list of all 234 BirdCLEF species from the mapping."""
    mapping = load_species_mapping(artifacts_dir)
    return sorted(mapping["birdclef_species"])


def build_binary_label_matrix(
    df: pl.DataFrame, species_list: list[str]
) -> np.ndarray:
    """Convert ground_truth semicolon-separated string to binary matrix.

    Args:
        df: DataFrame with 'ground_truth' column (semicolon-separated species IDs).
        species_list: Ordered list of species to include as columns.

    Returns:
        (N, len(species_list)) binary int8 matrix.
    """
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    n = len(df)
    k = len(species_list)
    matrix = np.zeros((n, k), dtype=np.int8)

    gt_col = df["ground_truth"].to_list()
    for i, gt in enumerate(gt_col):
        if not gt:
            continue
        for sp in str(gt).split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                matrix[i, species_to_idx[sp]] = 1

    return matrix


def load_focal_species(species: str, features_dir: str = FEATURES_DIR) -> pl.DataFrame:
    """Load a single species' focal data from the partitioned parquet."""
    path = Path(features_dir).expanduser() / "focal" / "by_species" / f"{species}.parquet"
    return pl.read_parquet(path)


def load_site(site: str, features_dir: str = FEATURES_DIR) -> pl.DataFrame:
    """Load a single site's unlabeled soundscape data from the partitioned parquet."""
    path = Path(features_dir).expanduser() / "soundscape_unlabeled" / "by_site" / f"{site}.parquet"
    return pl.read_parquet(path)


def list_focal_species(features_dir: str = FEATURES_DIR) -> list[str]:
    """List species that have partitioned focal parquets."""
    d = Path(features_dir).expanduser() / "focal" / "by_species"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def list_sites(features_dir: str = FEATURES_DIR) -> list[str]:
    """List sites that have partitioned unlabeled parquets."""
    d = Path(features_dir).expanduser() / "soundscape_unlabeled" / "by_site"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def embeddings_to_numpy(df: pl.DataFrame) -> np.ndarray:
    """Convert polars List(Float32) 'embedding' column to (N, 1536) numpy array."""
    return np.stack(df["embedding"].to_list()).astype(np.float32)


def logits_to_numpy(df: pl.DataFrame) -> np.ndarray:
    """Convert polars List(Float32) 'perch_logits' column to (N, 14795) numpy array."""
    return np.stack(df["perch_logits"].to_list()).astype(np.float32)
