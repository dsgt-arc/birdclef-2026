"""Tests for soundscape pretraining data loader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from birdclef_009.data.pretrain import SoundscapePretrainDataset


@pytest.fixture
def soundscape_dir():
    """Create a temporary directory with test soundscape files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sr = 32000
        # Create two 30s soundscape files
        for i in range(2):
            audio = np.random.randn(sr * 30).astype(np.float32) * 0.1
            sf.write(str(Path(tmpdir) / f"soundscape_{i}.wav"), audio, sr)
        yield tmpdir


def test_dataset_loads(soundscape_dir):
    """Dataset should load and index correctly."""
    ds = SoundscapePretrainDataset(
        soundscape_dir=soundscape_dir,
        sample_rate=32000,
        chunk_duration=10.0,
    )
    # 2 files × 30s / 10s = 6 chunks
    assert len(ds) == 6


def test_waveform_shape(soundscape_dir):
    """Each item should return waveform of correct shape."""
    ds = SoundscapePretrainDataset(
        soundscape_dir=soundscape_dir,
        sample_rate=32000,
        chunk_duration=10.0,
    )
    item = ds[0]
    assert "waveform" in item
    assert item["waveform"].shape == (320000,), f"Expected (320000,), got {item['waveform'].shape}"


def test_chunks_cover_full_file(soundscape_dir):
    """All chunks together should cover the full file duration."""
    ds = SoundscapePretrainDataset(
        soundscape_dir=soundscape_dir,
        sample_rate=32000,
        chunk_duration=10.0,
    )
    # Each 30s file produces 3 chunks
    file_chunks = [c for c in ds.chunks if "soundscape_0" in c[0]]
    assert len(file_chunks) == 3


def test_empty_dir():
    """Dataset should handle empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = SoundscapePretrainDataset(
            soundscape_dir=tmpdir,
            sample_rate=32000,
            chunk_duration=10.0,
        )
        assert len(ds) == 0
