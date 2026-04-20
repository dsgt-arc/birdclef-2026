import numpy as np
import pandas as pd
import pytest

from birdclef_kaggle.submission import build_submission


def _three_species():
    return ["a", "b", "c"]


def test_build_submission_basic(tmp_path):
    species = _three_species()
    row_ids = [f"BC2026_Test_{i:04d}_S01_20240101_120000_5" for i in range(4)]
    preds = np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0

    out = build_submission(row_ids, preds, species)
    assert list(out.columns) == ["row_id", "a", "b", "c"]
    assert len(out) == 4
    np.testing.assert_allclose(out[species].values, preds)


def test_build_submission_writes_csv(tmp_path):
    species = _three_species()
    row_ids = ["BC2026_Test_0001_S01_20240101_120000_5"]
    preds = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    output_path = tmp_path / "sub.csv"

    out = build_submission(row_ids, preds, species, output_path=output_path)
    assert output_path.exists()

    loaded = pd.read_csv(output_path)
    np.testing.assert_allclose(loaded[species].values, out[species].values)


def test_build_submission_reindexes_to_sample(tmp_path):
    species = _three_species()
    sample_path = tmp_path / "sample_submission.csv"
    sample = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "a": [1.0 / 3, 1.0 / 3, 1.0 / 3],
            "b": [1.0 / 3, 1.0 / 3, 1.0 / 3],
            "c": [1.0 / 3, 1.0 / 3, 1.0 / 3],
        }
    )
    sample.to_csv(sample_path, index=False)

    preds = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], dtype=np.float32)
    out = build_submission(
        row_ids=["r2", "r1"],
        predictions=preds,
        species=species,
        sample_submission_path=sample_path,
    )

    assert list(out["row_id"]) == ["r1", "r2", "r3"]
    np.testing.assert_allclose(out.loc[out.row_id == "r1", species].values[0], preds[1])
    np.testing.assert_allclose(out.loc[out.row_id == "r2", species].values[0], preds[0])
    np.testing.assert_allclose(
        out.loc[out.row_id == "r3", species].values[0],
        [1.0 / 3, 1.0 / 3, 1.0 / 3],
    )


def test_build_submission_shape_mismatch_raises():
    with pytest.raises(ValueError, match="predictions"):
        build_submission(["r"], np.zeros((2, 3), dtype=np.float32), _three_species())
    with pytest.raises(ValueError, match="species"):
        build_submission(["r"], np.zeros((1, 2), dtype=np.float32), _three_species())


def test_build_submission_rejects_mismatched_sample_species(tmp_path):
    species = _three_species()
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"row_id": ["r1"], "x": [0.5], "y": [0.5]}).to_csv(sample_path, index=False)

    with pytest.raises(ValueError, match="species set mismatch"):
        build_submission(
            row_ids=["r1"],
            predictions=np.zeros((1, 3), dtype=np.float32),
            species=species,
            sample_submission_path=sample_path,
        )
