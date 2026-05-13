"""Shared filesystem paths and read constants for experiment 200 analyses."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"

DATA_ROOT = Path("/mnt/data/scratch/birdclef-2026")
TRAIN_CSV = DATA_ROOT / "train.csv"
TAXONOMY_CSV = DATA_ROOT / "taxonomy.csv"
SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"
TRAIN_SOUNDSCAPES_LABELS_CSV = DATA_ROOT / "train_soundscapes_labels.csv"
RECORDING_LOCATION_TXT = DATA_ROOT / "recording_location.txt"
TRAIN_AUDIO_DIR = DATA_ROOT / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_ROOT / "train_soundscapes"
RAVEN_PRO_MP3_DIR = DATA_ROOT / "raven_pro_mp3"

PERCH_ROOT = DATA_ROOT / "inference" / "perch-v2-tf"
PERCH_TRAIN_AUDIO_EMBEDDINGS = PERCH_ROOT / "train_audio" / "embeddings.parquet"
PERCH_TRAIN_AUDIO_LOGITS = PERCH_ROOT / "train_audio" / "logits.parquet"
PERCH_SOUNDSCAPE_EMBEDDINGS = PERCH_ROOT / "train_soundscapes" / "embeddings.parquet"
PERCH_SOUNDSCAPE_LOGITS = PERCH_ROOT / "train_soundscapes" / "logits.parquet"
PERCH_234_PROBS = PERCH_ROOT / "train_soundscapes" / "perch_234_probs.parquet"
PERCH_234_RAW_PROBS = PERCH_ROOT / "train_soundscapes" / "perch_234_raw_probs.parquet"
PERCH_234_SCORES = PERCH_ROOT / "train_soundscapes" / "perch_234_scores.parquet"

# Polars schema overrides for reading train.csv. The raw CSV mixes integer-looking
# eBird codes with string labels; force these columns to Utf8 so joins work.
TRAIN_SCHEMA_OVERRIDES = {
    "primary_label": pl.Utf8,
    "secondary_labels": pl.Utf8,
}


def artifact_dir(name: str) -> Path:
    """Create and return an artifact subdirectory.

    `name` may contain slashes to nest under `artifacts/` (e.g. ``"marbled_frog/motif_boxes"``).
    """
    path = ARTIFACTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def fresh_subdir(parent: Path, name: str) -> Path:
    """Wipe and recreate a leaf subdir so re-runs don't leave stale files.

    Use for output directories whose file list depends on inputs that may shrink
    between runs (per-recording plots, per-cluster grids, per-species figures).
    Top-level summary files and fixed CSVs don't need this — overwrite is safe.
    """
    path = parent / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def recording_files_for_label(label: str) -> list[str]:
    """Return ``"<label>/<filename>"`` paths for every focal train recording of a label."""
    return [f"{label}/{path.name}" for path in sorted((TRAIN_AUDIO_DIR / label).glob("*.ogg"))]
