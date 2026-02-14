"""Tests for BirdCLEF labeled training data loader."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from birdclef_009.data.finetune import BirdCLEFTrainDataset, build_species_list


@pytest.fixture
def labeled_data():
    """Create temporary labeled dataset with audio files and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = Path(tmpdir) / "train_audio"
        audio_dir.mkdir()
        sr = 32000

        # Create test recordings
        species_list = ["species_a", "species_b", "species_c"]
        rows = []
        for i, sp in enumerate(species_list):
            sp_dir = audio_dir / sp
            sp_dir.mkdir()
            for j in range(2):
                duration = 15 + j * 5  # 15s and 20s
                audio = np.random.randn(sr * duration).astype(np.float32) * 0.1
                fname = f"{sp}/recording_{j}.wav"
                sf.write(str(audio_dir / fname), audio, sr)
                rows.append({
                    "filename": fname,
                    "primary_label": sp,
                    "secondary_labels": "[]",
                })

        metadata_csv = Path(tmpdir) / "train_metadata.csv"
        pd.DataFrame(rows).to_csv(metadata_csv, index=False)

        yield {
            "audio_dir": str(audio_dir),
            "metadata_csv": str(metadata_csv),
            "species_list": species_list,
        }


def test_build_species_list(labeled_data):
    """Should extract sorted unique species from metadata."""
    species = build_species_list(labeled_data["metadata_csv"])
    assert species == ["species_a", "species_b", "species_c"]


def test_dataset_loads(labeled_data):
    """Dataset should load and create chunks."""
    df = pd.read_csv(labeled_data["metadata_csv"])
    ds = BirdCLEFTrainDataset(
        audio_dir=labeled_data["audio_dir"],
        metadata_df=df,
        species_list=labeled_data["species_list"],
        sample_rate=32000,
        chunk_duration=10.0,
    )
    assert len(ds) > 0


def test_labels_are_valid(labeled_data):
    """Labels should be valid multi-hot vectors."""
    df = pd.read_csv(labeled_data["metadata_csv"])
    ds = BirdCLEFTrainDataset(
        audio_dir=labeled_data["audio_dir"],
        metadata_df=df,
        species_list=labeled_data["species_list"],
        sample_rate=32000,
        chunk_duration=10.0,
    )
    item = ds[0]

    assert "clip_labels" in item
    assert item["clip_labels"].shape == (3,)  # 3 species
    assert (item["clip_labels"] >= 0).all()
    assert (item["clip_labels"] <= 1).all()
    assert item["clip_labels"].sum() >= 1.0  # At least one species active


def test_num_tokens_and_real_tokens(labeled_data):
    """Dataset items should include num_tokens and real_tokens."""
    df = pd.read_csv(labeled_data["metadata_csv"])
    ds = BirdCLEFTrainDataset(
        audio_dir=labeled_data["audio_dir"],
        metadata_df=df,
        species_list=labeled_data["species_list"],
        sample_rate=32000,
        chunk_duration=10.0,
        num_tokens=1000,
    )
    item = ds[0]

    assert "num_tokens" in item
    assert item["num_tokens"] == 1000
    assert "real_tokens" in item
    assert 0 < item["real_tokens"] <= 1000


def test_waveform_shape(labeled_data):
    """Waveform should have correct shape."""
    df = pd.read_csv(labeled_data["metadata_csv"])
    ds = BirdCLEFTrainDataset(
        audio_dir=labeled_data["audio_dir"],
        metadata_df=df,
        species_list=labeled_data["species_list"],
        sample_rate=32000,
        chunk_duration=10.0,
    )
    item = ds[0]
    assert item["waveform"].shape == (320000,)
