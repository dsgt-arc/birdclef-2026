"""Tests for submission CSV formatting."""

import tempfile
from pathlib import Path

import pandas as pd

from birdclef_009.inference.submission import build_submission


def test_output_matches_sample_columns():
    """Output CSV should have same columns as sample_submission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample submission
        sample_path = str(Path(tmpdir) / "sample_submission.csv")
        sample_df = pd.DataFrame({
            "row_id": ["sc_5", "sc_10"],
            "sp_a": [0.0, 0.0],
            "sp_b": [0.0, 0.0],
        })
        sample_df.to_csv(sample_path, index=False)

        # Create predictions
        predictions = [
            {"row_id": "sc_5", "sp_a": 0.8, "sp_b": 0.2},
            {"row_id": "sc_10", "sp_a": 0.1, "sp_b": 0.9},
        ]

        output_path = str(Path(tmpdir) / "submission.csv")
        result = build_submission(predictions, sample_path, output_path)

        assert list(result.columns) == ["row_id", "sp_a", "sp_b"]
        assert len(result) == 2


def test_missing_predictions_filled_with_zero():
    """Missing row_ids in predictions should be filled with 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = str(Path(tmpdir) / "sample_submission.csv")
        sample_df = pd.DataFrame({
            "row_id": ["sc_5", "sc_10", "sc_15"],
            "sp_a": [0.0, 0.0, 0.0],
        })
        sample_df.to_csv(sample_path, index=False)

        # Only provide predictions for 2 of 3 rows
        predictions = [
            {"row_id": "sc_5", "sp_a": 0.8},
            {"row_id": "sc_10", "sp_a": 0.3},
        ]

        output_path = str(Path(tmpdir) / "submission.csv")
        result = build_submission(predictions, sample_path, output_path)

        assert len(result) == 3
        assert result.loc[result["row_id"] == "sc_15", "sp_a"].values[0] == 0.0


def test_output_csv_written():
    """Output CSV file should be written to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = str(Path(tmpdir) / "sample_submission.csv")
        pd.DataFrame({"row_id": ["sc_5"], "sp_a": [0.0]}).to_csv(sample_path, index=False)

        predictions = [{"row_id": "sc_5", "sp_a": 0.5}]
        output_path = str(Path(tmpdir) / "output" / "submission.csv")

        build_submission(predictions, sample_path, output_path)

        assert Path(output_path).exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 1
