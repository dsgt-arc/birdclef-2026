"""Smoke tests for BamBird ROI extraction."""

import json

import pandas as pd
import pytest

from tests.conftest import requires_data


@requires_data
class TestBamBirdRunner:
    def test_extract_rois_returns_list(self, sample_focal_file):
        """extract_rois returns a list of ROI dicts."""
        from teacher_labels.bambird_runner import extract_rois

        events = extract_rois(sample_focal_file)
        assert isinstance(events, list)
        # May be empty for some files, but should not error
        for e in events:
            assert "min_t" in e
            assert "max_t" in e
            assert "min_f" in e
            assert "max_f" in e
            assert "duration_s" in e
            assert e["max_t"] > e["min_t"]

    def test_extract_rois_respects_duration_filter(self, sample_focal_file):
        """ROIs outside min/max duration are filtered out."""
        from teacher_labels.bambird_runner import extract_rois

        events = extract_rois(sample_focal_file, min_duration=0.0, max_duration=100.0)
        strict_events = extract_rois(sample_focal_file, min_duration=0.5, max_duration=2.0)
        # Strict filter should return <= events
        assert len(strict_events) <= len(events)

    def test_process_partition_schema(self, sample_focal_files, tmp_output):
        """process_partition produces correct parquet schema."""
        from teacher_labels.bambird_runner import process_partition

        output_path = tmp_output / "test_bambird.parquet"
        df = process_partition(sample_focal_files, output_path)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"file", "events_json", "num_rois", "duration_s"}
        assert len(df) == len(sample_focal_files)
        assert output_path.exists()

        # Verify events_json is valid JSON
        for events_str in df["events_json"]:
            events = json.loads(events_str)
            assert isinstance(events, list)

        # num_rois should be non-negative
        assert (df["num_rois"] >= 0).all()
        # duration_s should be positive
        assert (df["duration_s"] >= 0).all()
