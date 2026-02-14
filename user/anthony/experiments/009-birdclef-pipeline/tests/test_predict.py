"""Tests for inference prediction pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf
import torch

from birdclef_009.inference.predict import predict_soundscape


def _make_mock_model(num_classes):
    """Create a mock model with forward_windows returning (1, 2, C) logits."""
    model = MagicMock()
    model.eval = MagicMock()
    model.forward_windows = MagicMock(
        return_value=torch.randn(1, 2, num_classes)
    )
    return model


def test_10s_produces_two_5s_rows():
    """A 10s soundscape should produce exactly 2 prediction rows."""
    sr = 32000
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = np.random.randn(sr * 10).astype(np.float32) * 0.1
        audio_path = str(Path(tmpdir) / "test_soundscape.wav")
        sf.write(audio_path, audio, sr)

        species_list = ["sp_a", "sp_b", "sp_c"]
        model = _make_mock_model(3)

        predictions = predict_soundscape(
            model, audio_path, species_list,
            sample_rate=sr, chunk_duration=10.0,
        )

        assert len(predictions) == 2


def test_row_id_format():
    """Row IDs should be {soundscape_id}_{end_time}."""
    sr = 32000
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = np.random.randn(sr * 10).astype(np.float32) * 0.1
        audio_path = str(Path(tmpdir) / "XC123456.wav")
        sf.write(audio_path, audio, sr)

        species_list = ["sp_a"]
        model = _make_mock_model(1)

        predictions = predict_soundscape(
            model, audio_path, species_list,
            sample_rate=sr, chunk_duration=10.0,
        )

        assert predictions[0]["row_id"] == "XC123456_5"
        assert predictions[1]["row_id"] == "XC123456_10"


def test_30s_produces_six_rows():
    """A 30s soundscape should produce 6 prediction rows (3 chunks x 2 halves)."""
    sr = 32000
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = np.random.randn(sr * 30).astype(np.float32) * 0.1
        audio_path = str(Path(tmpdir) / "soundscape_001.wav")
        sf.write(audio_path, audio, sr)

        species_list = ["sp_a", "sp_b"]
        model = _make_mock_model(2)

        predictions = predict_soundscape(
            model, audio_path, species_list,
            sample_rate=sr, chunk_duration=10.0,
        )

        assert len(predictions) == 6
        expected_times = [5, 10, 15, 20, 25, 30]
        for pred, t in zip(predictions, expected_times):
            assert pred["row_id"] == f"soundscape_001_{t}"


def test_predictions_have_species_columns():
    """Each prediction row should have all species as keys."""
    sr = 32000
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = np.random.randn(sr * 10).astype(np.float32) * 0.1
        audio_path = str(Path(tmpdir) / "test.wav")
        sf.write(audio_path, audio, sr)

        species_list = ["sp_a", "sp_b", "sp_c"]
        model = _make_mock_model(3)

        predictions = predict_soundscape(
            model, audio_path, species_list,
            sample_rate=sr, chunk_duration=10.0,
        )

        for pred in predictions:
            assert "row_id" in pred
            for sp in species_list:
                assert sp in pred
                assert isinstance(pred[sp], float)
