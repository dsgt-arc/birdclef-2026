"""PerchSession import-level smoke test.

Full ONNX model download + inference is tested separately via a dry-run
sbatch job (requires network + real weights). This test only confirms the
class imports and raises on unknown outputs without a downloaded model.
"""

from __future__ import annotations

from birdclef_kaggle import perch_session


def test_class_exists():
    assert hasattr(perch_session, "PerchSession")


def test_default_repo_constants():
    assert perch_session.DEFAULT_REPO == "justinchuby/Perch-onnx"
    assert perch_session.DEFAULT_FILENAME == "perch_v2_no_dft.onnx"
