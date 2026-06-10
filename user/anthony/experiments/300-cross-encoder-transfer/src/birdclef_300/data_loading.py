"""Unified encoder-agnostic embedding cache: schema, paths, and probe helpers.

This is the decoupling contract between exp 300 (the eval hub) and exp 301 (the
adaptation spoke). Every encoder — whatever its backend — writes one parquet per
(encoder, variant) with these columns:

    filename       str            audio filename
    window_idx     int            0-based window within the file
    start_sec      float          window start in seconds
    source         str            "soundscape-labeled" (the only source 300 needs)
    split          str            "train" / "val" / ""
    primary_label  str            ground-truth primary species (or "")
    ground_truth   str            semicolon-joined species IDs (or "")
    encoder        str            "beats" | "ast" | "eat" | "ssamba_hf" | "perch" | "birdnet"
    variant        str            "frozen" | "lora-r8" | "dapt-ep10" | ...
    embed_dim      int            embedding width (lets variable-width encoders coexist)
    embedding      list[float32]  the pooled embedding

Site/label helpers mirror exp 106's `data_loading` so the ridge probe is a clean
port. Species list is derived from the labeled parquet's own ground_truth, so
exp 300 carries no dependency on exp 106's perch_species_mapping.json.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

# Filename pattern: BC2026_Train_{id}_{site}_{date}_{time}.ogg
_SITE_RE = re.compile(r"_(S\d+)_")

SCHEMA_COLUMNS = [
    "filename", "window_idx", "start_sec", "source", "split",
    "primary_label", "ground_truth", "encoder", "variant", "embed_dim", "embedding",
]


def labeled_parquet_path(base_dir: str, encoder: str, variant: str) -> Path:
    """Partitioned cache path for one (encoder, variant) on the labeled source."""
    return (
        Path(base_dir).expanduser()
        / "embeddings"
        / f"encoder={encoder}"
        / f"variant={variant}"
        / "source=soundscape-labeled"
        / "data.parquet"
    )


def load_encoder_labeled(base_dir: str, encoder: str, variant: str = "frozen") -> pl.DataFrame:
    """Read one encoder/variant's labeled-soundscape cache."""
    return pl.read_parquet(labeled_parquet_path(base_dir, encoder, variant))


def extract_site(filename: str) -> str:
    """Extract site ID (e.g. 'S08') from a soundscape filename."""
    m = _SITE_RE.search(filename)
    return m.group(1) if m else "unknown"


def embeddings_to_numpy(df: pl.DataFrame) -> np.ndarray:
    """Convert the polars List(Float32) 'embedding' column to (N, D) numpy."""
    return np.stack(df["embedding"].to_list()).astype(np.float32)


def build_species_list(df: pl.DataFrame) -> list[str]:
    """Sorted unique species appearing in the labeled parquet's ground_truth."""
    species: set[str] = set()
    for gt in df["ground_truth"].to_list():
        if not gt:
            continue
        for sp in str(gt).split(";"):
            sp = sp.strip()
            if sp:
                species.add(sp)
    return sorted(species)


def build_binary_label_matrix(df: pl.DataFrame, species_list: list[str]) -> np.ndarray:
    """Convert ground_truth (semicolon-separated) to an (N, K) int8 binary matrix."""
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    matrix = np.zeros((len(df), len(species_list)), dtype=np.int8)
    for i, gt in enumerate(df["ground_truth"].to_list()):
        if not gt:
            continue
        for sp in str(gt).split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                matrix[i, species_to_idx[sp]] = 1
    return matrix
