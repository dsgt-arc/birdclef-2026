"""Shared test fixtures and skip markers."""

from pathlib import Path

import pytest

DATA_ROOT = Path("~/scratch/birdclef-2026/data/birdclef-2025").expanduser()
FOCAL_DIR = DATA_ROOT / "train_audio"
SOUNDSCAPE_DIR = DATA_ROOT / "train_soundscapes"

PERCH_TFLITE = Path("~/scratch/birdclef-2026/outputs/exp010/perch.tflite").expanduser()

has_data = FOCAL_DIR.exists() and any(FOCAL_DIR.iterdir())
requires_data = pytest.mark.skipif(not has_data, reason="BirdCLEF 2025 data not available")
requires_perch_tflite = pytest.mark.skipif(
    not PERCH_TFLITE.exists(),
    reason=f"Perch TFLite not compiled (run teacher-compile-perch first): {PERCH_TFLITE}",
)


@pytest.fixture
def sample_focal_file():
    """Return path to a single focal recording for testing."""
    files = sorted(FOCAL_DIR.rglob("*.ogg"))
    if not files:
        pytest.skip("No focal recordings found")
    return files[0]


@pytest.fixture
def sample_soundscape_file():
    """Return path to a single soundscape recording for testing."""
    files = sorted(SOUNDSCAPE_DIR.glob("*.ogg"))
    if not files:
        pytest.skip("No soundscape recordings found")
    return files[0]


@pytest.fixture
def sample_focal_files():
    """Return a small batch of focal recordings for batch tests."""
    files = sorted(FOCAL_DIR.rglob("*.ogg"))[:3]
    if not files:
        pytest.skip("No focal recordings found")
    return files


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temporary output directory."""
    return tmp_path / "output"
