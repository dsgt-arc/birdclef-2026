"""Smoke tests for BirdNET inference."""

import pandas as pd
import pytest

from tests.conftest import requires_data


@requires_data
class TestBirdNETRunner:
    def test_embed_produces_two_dataframes(self, sample_focal_files):
        """BirdNET model.embed() returns (embed_df, predict_df)."""
        import bioacoustics_model_zoo as bmz

        model = bmz.list_models()["BirdNET"]()
        embed_df, predict_df = model.embed(
            [p.as_posix() for p in sample_focal_files],
            return_preds=True,
            clip_step=1.0,
        )

        assert isinstance(embed_df, pd.DataFrame)
        assert isinstance(predict_df, pd.DataFrame)
        # Should have rows (clips across all files)
        assert len(embed_df) > 0
        assert len(predict_df) > 0
        # Embed and predict should have same number of rows
        assert len(embed_df) == len(predict_df)

    def test_embed_schema(self, sample_focal_files):
        """Verify embedding and prediction column structure."""
        import bioacoustics_model_zoo as bmz

        model = bmz.list_models()["BirdNET"]()
        embed_df, predict_df = model.embed(
            [p.as_posix() for p in sample_focal_files[:1]],
            return_preds=True,
            clip_step=1.0,
        )

        # After reset_index, should have file/start_time/end_time + feature cols
        embed_reset = embed_df.reset_index()
        predict_reset = predict_df.reset_index()

        # Check metadata columns exist
        for col in ["file", "start_time", "end_time"]:
            assert col in embed_reset.columns or col in embed_reset.index.names, \
                f"Missing {col} in embed columns: {embed_reset.columns.tolist()[:10]}"
            assert col in predict_reset.columns or col in predict_reset.index.names, \
                f"Missing {col} in predict columns: {predict_reset.columns.tolist()[:10]}"

    def test_parquet_roundtrip(self, sample_focal_files, tmp_output):
        """Verify embed/predict survive parquet save/load."""
        import bioacoustics_model_zoo as bmz

        model = bmz.list_models()["BirdNET"]()
        embed_df, predict_df = model.embed(
            [p.as_posix() for p in sample_focal_files[:1]],
            return_preds=True,
            clip_step=1.0,
        )

        tmp_output.mkdir(parents=True, exist_ok=True)
        embed_path = tmp_output / "embed.parquet"
        predict_path = tmp_output / "predict.parquet"

        embed_df.reset_index().to_parquet(embed_path, index=False)
        predict_df.reset_index().to_parquet(predict_path, index=False)

        # Read back and verify shapes match
        embed_loaded = pd.read_parquet(embed_path)
        predict_loaded = pd.read_parquet(predict_path)

        assert embed_loaded.shape[0] == len(embed_df)
        assert predict_loaded.shape[0] == len(predict_df)
