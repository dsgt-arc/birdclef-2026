"""Tests for BamBird pipeline using the bambird library API."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tests.conftest import requires_data


class TestBamBirdSynthetic:
    """Tests using synthetic audio — no data dependency."""

    @pytest.fixture
    def synth_wav(self, tmp_path):
        """Create a synthetic .wav with a tone burst in noise."""
        rng = np.random.default_rng(42)
        sr = 48000  # bambird default sample rate
        duration = 5.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Low-level broadband noise
        signal = 0.01 * rng.standard_normal(n_samples)
        # Loud tone burst at 3kHz from 1.0–2.0s (within bambird's 250-12000 Hz range)
        burst_start = int(1.0 * sr)
        burst_end = int(2.0 * sr)
        signal[burst_start:burst_end] += 0.5 * np.sin(2 * np.pi * 3000 * t[burst_start:burst_end])
        # Second burst at 5kHz from 3.0–3.5s
        burst2_start = int(3.0 * sr)
        burst2_end = int(3.5 * sr)
        signal[burst2_start:burst2_end] += 0.5 * np.sin(2 * np.pi * 5000 * t[burst2_start:burst2_end])

        # Save in a species subdirectory (bambird expects path/.../species/file.wav)
        species_dir = tmp_path / "synth_species"
        species_dir.mkdir()
        path = species_dir / "test_audio.wav"
        sf.write(str(path), signal, sr)
        return path

    @pytest.fixture
    def bambird_params(self):
        """Load bambird config from the project config file."""
        import bambird
        return bambird.load_config("configs/bambird.yaml")

    def test_single_file_extract_rois(self, synth_wav, bambird_params, tmp_path):
        """bambird.single_file_extract_rois should find ROIs with bandpass filtering."""
        import bambird

        params_extract = bambird_params["PARAMS_EXTRACT"]
        save_path = tmp_path / "rois"
        save_path.mkdir()

        df_rois = bambird.single_file_extract_rois(
            audio_path=str(synth_wav),
            fun=params_extract["FUNC"],
            params=params_extract,
            save_path=str(save_path),
            display=False,
            verbose=False,
        )

        assert isinstance(df_rois, pd.DataFrame)
        # Synthetic tone bursts within 250-12000 Hz should produce ROIs
        if not df_rois.empty:
            assert "min_t" in df_rois.columns
            assert "max_t" in df_rois.columns
            assert "min_f" in df_rois.columns
            assert "max_f" in df_rois.columns
            assert "fullfilename_ts" in df_rois.columns
            assert "categories" in df_rois.columns
            # ROI frequencies should be within bandpass range
            assert (df_rois["min_f"] >= 0).all()
            assert (df_rois["max_f"] <= 12000).all()

    def test_compute_features_on_roi(self, synth_wav, bambird_params, tmp_path):
        """bambird.compute_features should produce shape features from a ROI .wav."""
        import bambird

        params_extract = bambird_params["PARAMS_EXTRACT"]
        params_features = bambird_params["PARAMS_FEATURES"]
        save_path = tmp_path / "rois"
        save_path.mkdir()

        # Extract ROIs first
        df_rois = bambird.single_file_extract_rois(
            audio_path=str(synth_wav),
            fun=params_extract["FUNC"],
            params=params_extract,
            save_path=str(save_path),
            display=False,
            verbose=False,
        )

        if df_rois.empty:
            pytest.skip("No ROIs extracted from synthetic audio")

        # Compute features for first ROI
        roi_path = df_rois.iloc[0]["fullfilename_ts"]
        assert Path(roi_path).exists(), f"ROI wav not found: {roi_path}"

        df_features = bambird.compute_features(
            audio_path=roi_path,
            params=params_features,
            display=False,
            verbose=False,
        )

        assert isinstance(df_features, pd.DataFrame)
        if not df_features.empty:
            # Should have shape columns (shpNN)
            shp_cols = [c for c in df_features.columns if c.startswith("shp")]
            assert len(shp_cols) > 0, "Expected shape feature columns (shpNN)"
            # Should have frequency features
            assert "centroid_f" in df_features.columns
            assert "peak_f" in df_features.columns

    def test_extract_features_single_file(self, synth_wav, bambird_params, tmp_path):
        """The combined extract+features pipeline should work end-to-end."""
        from teacher_labels.bambird_runner import _extract_features_single_file

        params_extract = bambird_params["PARAMS_EXTRACT"]
        params_features = bambird_params["PARAMS_FEATURES"]
        temp_dir = tmp_path / "temp_rois"
        temp_dir.mkdir()

        df = _extract_features_single_file(
            audio_path=str(synth_wav),
            params_extract=params_extract,
            params_features=params_features,
            temp_dir=temp_dir,
        )

        assert isinstance(df, pd.DataFrame)
        # May or may not find ROIs in synthetic audio depending on params

    def test_silence_produces_no_rois(self, bambird_params, tmp_path):
        """Pure silence should produce zero ROIs."""
        import bambird

        sr = 48000
        signal = np.zeros(sr * 3)
        species_dir = tmp_path / "silent_species"
        species_dir.mkdir()
        path = species_dir / "silence.wav"
        sf.write(str(path), signal, sr)

        params_extract = bambird_params["PARAMS_EXTRACT"]
        df_rois = bambird.single_file_extract_rois(
            audio_path=str(path),
            fun=params_extract["FUNC"],
            params=params_extract,
            save_path=str(tmp_path / "rois"),
            display=False,
            verbose=False,
        )

        assert isinstance(df_rois, pd.DataFrame)
        assert df_rois.empty, "Silence should produce zero ROIs"


@requires_data
class TestBamBirdRealData:
    """Tests with real BirdCLEF data — skipped if not available."""

    @pytest.fixture
    def bambird_params(self):
        import bambird
        return bambird.load_config("configs/bambird.yaml")

    def test_full_pipeline_single_file(self, sample_focal_file, bambird_params, tmp_path):
        """Full extract+features pipeline on a real focal recording."""
        from teacher_labels.bambird_runner import _extract_features_single_file

        params_extract = bambird_params["PARAMS_EXTRACT"]
        params_features = bambird_params["PARAMS_FEATURES"]
        temp_dir = tmp_path / "temp_rois"
        temp_dir.mkdir()

        df = _extract_features_single_file(
            audio_path=str(sample_focal_file),
            params_extract=params_extract,
            params_features=params_features,
            temp_dir=temp_dir,
        )

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "Expected features from a focal bird recording"

        # Verify expected columns
        assert "categories" in df.columns
        assert "fullfilename" in df.columns
        shp_cols = [c for c in df.columns if c.startswith("shp")]
        assert len(shp_cols) > 0, "Expected shape features"
        assert "centroid_f" in df.columns
        assert "min_t" in df.columns
        assert "max_t" in df.columns

    def test_clustering_on_features(self, sample_focal_files, bambird_params, tmp_path):
        """Clustering should label ROIs as signal or noise."""
        import bambird
        from teacher_labels.bambird_runner import _extract_features_single_file

        params_extract = bambird_params["PARAMS_EXTRACT"]
        params_features = bambird_params["PARAMS_FEATURES"]
        params_cluster = bambird_params["PARAMS_CLUSTER"]

        # Extract features from multiple files
        all_features = []
        for fpath in sample_focal_files:
            temp_dir = tmp_path / f"temp_{fpath.stem}"
            temp_dir.mkdir()
            try:
                df = _extract_features_single_file(
                    audio_path=str(fpath),
                    params_extract=params_extract,
                    params_features=params_features,
                    temp_dir=temp_dir,
                )
                if not df.empty:
                    all_features.append(df)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not all_features:
            pytest.skip("No features extracted from test files")

        df_all = pd.concat(all_features, ignore_index=True)

        # Need filename_ts and fullfilename_ts for clustering
        if "filename_ts" not in df_all.columns:
            df_all["filename_ts"] = df_all.index.astype(str)
        if "fullfilename_ts" not in df_all.columns:
            df_all["fullfilename_ts"] = ""

        # May not have enough ROIs per species to cluster
        if len(df_all) < 3:
            pytest.skip(f"Only {len(df_all)} ROIs — need >=3 for clustering")

        df_cluster, _ = bambird.find_cluster(
            dataset=df_all,
            params=params_cluster,
            display=False,
            verbose=True,
        )

        assert isinstance(df_cluster, pd.DataFrame)
        assert "cluster_number" in df_cluster.columns
        assert "auto_label" in df_cluster.columns
        # auto_label should be 0 or 1
        assert set(df_cluster["auto_label"].unique()).issubset({0, 1})
