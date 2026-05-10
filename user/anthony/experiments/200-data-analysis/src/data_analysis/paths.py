"""Shared filesystem paths for experiment 200 analyses."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
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


def artifact_dir(name: str) -> Path:
    """Create and return an artifact subdirectory."""
    path = ARTIFACTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
