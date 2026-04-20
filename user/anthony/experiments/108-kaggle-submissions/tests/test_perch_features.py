"""Verify birdclef_kaggle.perch_features matches birdclef_107.perch_features
on matched inputs. Parity test guards against drift when either side moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdclef_kaggle.perch_features import PerchSidecarProjector


def _make_logits(seed: int = 0, n: int = 5, raw_dim: int = 14795) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, raw_dim)).astype(np.float32) * 3.0


def test_shapes_and_dtypes():
    proj = PerchSidecarProjector()
    logits = _make_logits(n=7)
    out = proj.project(logits)
    assert out["mapped_probs"].shape == (7, 234)
    assert out["mapped_probs"].dtype == np.float32
    assert out["topk_idx"].shape == (7, 64)
    assert out["topk_idx"].dtype == np.int32
    assert out["topk_val"].shape == (7, 64)
    assert out["topk_val"].dtype == np.float32


def test_mapped_probs_are_sigmoid_of_logits():
    proj = PerchSidecarProjector()
    logits = _make_logits(n=3)
    out = proj.project(logits)
    # Every direct-mapped species probability equals sigmoid(logit).
    direct_mask = proj.direct_idx >= 0
    direct_species = np.flatnonzero(direct_mask)
    raw_cols = proj.direct_idx[direct_mask]
    expected = 1.0 / (1.0 + np.exp(-logits[:, raw_cols]))
    np.testing.assert_allclose(
        out["mapped_probs"][:, direct_species], expected, rtol=1e-5, atol=1e-6
    )


def test_topk_sorted_descending_and_excludes_used():
    proj = PerchSidecarProjector()
    logits = _make_logits(n=4)
    out = proj.project(logits)
    # Descending by value.
    for row_vals in out["topk_val"]:
        assert np.all(np.diff(row_vals) <= 1e-6)
    # Indices never coincide with mapped-used raw indices.
    used = set(proj.used_raw_indices.tolist())
    for row_idx in out["topk_idx"]:
        for i in row_idx:
            if i >= 0:
                assert i not in used


def test_matches_birdclef_107_projector():
    """Parity with the authoritative exp 107 implementation."""
    try:
        from birdclef_107.perch_features import (
            PerchSidecarProjector as RefProjector,
        )
    except ImportError:
        pytest.skip("birdclef_107 not importable in this env")

    ref = RefProjector()
    ours = PerchSidecarProjector()
    logits = _make_logits(n=6)

    ref_out = ref.project(logits)
    our_out = ours.project(logits)

    np.testing.assert_allclose(
        our_out["mapped_probs"], ref_out["mapped_probs"], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(our_out["topk_idx"], ref_out["topk_idx"])
    np.testing.assert_allclose(
        our_out["topk_val"], ref_out["topk_val"], rtol=1e-6, atol=1e-7
    )
