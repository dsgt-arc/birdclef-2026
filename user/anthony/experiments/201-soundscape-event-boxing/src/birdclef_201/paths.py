"""Canonical data paths + artifact helpers for experiment 201 (standalone)."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/mnt/data/scratch/birdclef-2026")
TAXONOMY_CSV = DATA_ROOT / "taxonomy.csv"
TRAIN_SOUNDSCAPES_LABELS_CSV = DATA_ROOT / "train_soundscapes_labels.csv"
TRAIN_SOUNDSCAPES_DIR = DATA_ROOT / "train_soundscapes"
PERCH_SOUNDSCAPE_EMBEDDINGS = (
    DATA_ROOT / "inference/perch-v2-tf/train_soundscapes/embeddings.parquet"
)

# src/birdclef_201/paths.py -> parents[2] is the experiment root.
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"

# Derived clips / spectrogram PNGs for Label Studio live on scratch (already a
# serving root) — never in the experiment dir. Keeps git clean; regenerable.
SCRATCH_CLIPS_DIR = Path("/mnt/data/scratch/birdclef-201-clips")


def artifact_dir(name: str) -> Path:
    """Create and return ``artifacts/<name>``."""
    d = ARTIFACTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
