"""Smoke tests for Perch inference via TFLite."""

import pandas as pd
import pytest

from tests.conftest import PERCH_TFLITE, requires_data, requires_perch_tflite


@requires_data
@requires_perch_tflite
class TestPerchTFLite:
    def test_tflite_produces_embeddings(self, sample_focal_files):
        """TFLite interpreter produces 1280-dim embeddings."""
        from teacher_labels.perch_runner import _run_tflite

        elapsed, embed_df, predict_df = _run_tflite(
            PERCH_TFLITE,
            [p.as_posix() for p in sample_focal_files[:1]],
            clip_step=5.0,
        )

        assert isinstance(embed_df, pd.DataFrame)
        assert isinstance(predict_df, pd.DataFrame)
        assert len(embed_df) > 0
        assert len(predict_df) > 0
        assert elapsed > 0

    def test_tflite_embed_dimensions(self, sample_focal_files):
        """TFLite embeddings should be 1280-dim."""
        from teacher_labels.perch_runner import _run_tflite

        elapsed, embed_df, predict_df = _run_tflite(
            PERCH_TFLITE,
            [p.as_posix() for p in sample_focal_files[:1]],
            clip_step=5.0,
        )

        embed_reset = embed_df.reset_index()
        meta_cols = {"file", "start_time", "end_time"}
        feature_cols = [c for c in embed_reset.columns if c not in meta_cols]
        assert len(feature_cols) == 1280, \
            f"Expected 1280 feature dims, got {len(feature_cols)}"

    def test_tflite_parquet_roundtrip(self, sample_focal_files, tmp_output):
        """Verify TFLite embed/predict survive parquet save/load."""
        from teacher_labels.perch_runner import _run_tflite

        elapsed, embed_df, predict_df = _run_tflite(
            PERCH_TFLITE,
            [p.as_posix() for p in sample_focal_files[:1]],
            clip_step=5.0,
        )

        tmp_output.mkdir(parents=True, exist_ok=True)
        embed_path = tmp_output / "embed.parquet"
        predict_path = tmp_output / "predict.parquet"

        embed_df.reset_index().to_parquet(embed_path, index=False)
        predict_df.reset_index().to_parquet(predict_path, index=False)

        embed_loaded = pd.read_parquet(embed_path)
        predict_loaded = pd.read_parquet(predict_path)

        assert embed_loaded.shape[0] == len(embed_df)
        assert predict_loaded.shape[0] == len(predict_df)


@requires_data
def test_runner_fails_without_tflite(sample_focal_files, tmp_path):
    """Perch runner must fail loudly if TFLite model is missing."""
    import yaml
    from teacher_labels.perch_runner import run

    # Write a config pointing to a nonexistent tflite path
    cfg = {
        "data": {"focal_dir": str(sample_focal_files[0].parent.parent)},
        "output": {"root": str(tmp_path / "output")},
        "partitions": {"focal": 1},
        "models": {"perch": {
            "tflite_path": str(tmp_path / "does_not_exist.tflite"),
            "clip_step": 5.0,
        }},
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    with pytest.raises(FileNotFoundError, match="teacher-compile-perch"):
        run(config=str(config_path), source="focal", partition_id=0)
