"""Shared utilities for teacher label generation."""

from pathlib import Path

import yaml


def load_config(config_path: str = "configs/teacher.yaml") -> dict:
    """Load the teacher config YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_audio_files(directory: str, source: str) -> list[Path]:
    """List all audio files in a directory.

    Args:
        directory: Root directory containing audio files.
        source: Either "focal" or "soundscape".

    Returns:
        Sorted list of audio file paths.
    """
    root = Path(directory).expanduser()
    if source == "focal":
        # Focal recordings are in species subdirectories
        files = sorted(root.rglob("*.ogg"))
    elif source == "soundscape":
        # Soundscapes are flat .ogg files
        files = sorted(root.glob("*.ogg"))
    else:
        raise ValueError(f"Unknown source: {source!r}, expected 'focal' or 'soundscape'")
    return files


def partition_files(files: list[Path], num_partitions: int, partition_id: int) -> list[Path]:
    """Select files for a given partition using modular arithmetic.

    Args:
        files: Full sorted list of audio files.
        num_partitions: Total number of partitions.
        partition_id: This partition's ID (0-indexed).

    Returns:
        Subset of files belonging to this partition.
    """
    return [f for i, f in enumerate(files) if i % num_partitions == partition_id]


def ensure_output_dir(path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
