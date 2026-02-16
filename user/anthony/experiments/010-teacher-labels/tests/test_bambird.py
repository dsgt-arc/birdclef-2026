"""Tests for BamBird ROI extraction."""

import json

import numpy as np
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
        for e in events:
            assert "min_t" in e
            assert "max_t" in e
            assert "min_f" in e
            assert "max_f" in e
            assert "duration_s" in e
            assert e["max_t"] > e["min_t"]

    def test_extract_rois_finds_events(self, sample_focal_file):
        """extract_rois should find at least some ROIs in a focal bird recording."""
        from teacher_labels.bambird_runner import extract_rois

        events = extract_rois(sample_focal_file)
        assert len(events) > 0, "Expected ROIs in a focal bird recording"

    def test_extract_rois_physical_coordinates(self, sample_focal_file):
        """ROI coordinates should be in physical units (seconds, Hz), not pixels."""
        import soundfile as sf
        from teacher_labels.bambird_runner import extract_rois

        info = sf.info(str(sample_focal_file))
        events = extract_rois(sample_focal_file, min_duration=0.0, max_duration=100.0)
        assert len(events) > 0, "Need at least one ROI for this test"

        for e in events:
            # Times should be within the file duration
            assert 0.0 <= e["min_t"] <= info.duration, (
                f"min_t={e['min_t']} outside [0, {info.duration}]"
            )
            assert 0.0 <= e["max_t"] <= info.duration + 0.1, (
                f"max_t={e['max_t']} outside [0, {info.duration}]"
            )
            # Frequencies should be in audible range (Hz), not pixel indices
            assert e["min_f"] >= 0.0
            assert e["max_f"] <= info.samplerate / 2 + 1
            assert e["max_f"] > 100.0, (
                f"max_f={e['max_f']} suspiciously low — likely pixel coords, not Hz"
            )

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

    def test_process_partition_finds_rois(self, sample_focal_files, tmp_output):
        """process_partition should detect ROIs across a batch of focal recordings."""
        from teacher_labels.bambird_runner import process_partition

        output_path = tmp_output / "test_bambird_rois.parquet"
        df = process_partition(sample_focal_files, output_path)

        total_rois = df["num_rois"].sum()
        assert total_rois > 0, (
            f"Zero ROIs across {len(sample_focal_files)} focal files — "
            "likely a bug in extract_rois()"
        )


class TestExtractRoisSynthetic:
    """Tests using synthetic audio — no data dependency."""

    def test_tone_in_noise_produces_rois(self, tmp_path):
        """A loud tone burst in background noise should produce at least one ROI.

        median_equalizer needs broadband noise to compute meaningful medians;
        pure silence gives degenerate results.
        """
        import soundfile as sf
        from teacher_labels.bambird_runner import extract_rois

        rng = np.random.default_rng(42)
        sr = 22050
        duration = 5.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Low-level broadband noise as background
        signal = 0.01 * rng.standard_normal(n_samples)
        # Loud tone burst at 2kHz from 2.0–2.5s
        burst_start = int(2.0 * sr)
        burst_end = int(2.5 * sr)
        signal[burst_start:burst_end] += 0.5 * np.sin(2 * np.pi * 2000 * t[burst_start:burst_end])

        path = tmp_path / "burst_in_noise.wav"
        sf.write(str(path), signal, sr)

        events = extract_rois(path, min_duration=0.01, max_duration=10.0)
        assert len(events) > 0, "A loud tone burst in noise should produce at least one ROI"
        for e in events:
            assert e["min_f"] > 0
            assert e["max_f"] <= sr / 2

    def test_silence_produces_no_rois(self, tmp_path):
        """Pure silence should produce zero ROIs without crashing."""
        import soundfile as sf
        from teacher_labels.bambird_runner import extract_rois

        sr = 22050
        signal = np.zeros(sr * 2)  # 2 seconds of silence
        path = tmp_path / "silence.wav"
        sf.write(str(path), signal, sr)

        events = extract_rois(path)
        assert isinstance(events, list)
        assert len(events) == 0
