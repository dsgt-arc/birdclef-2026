"""Exp 036 Part 3 — ClassBalancedMixSamp (Sydorskyi class-balanced focal draws).

Verifies it preserves MixSamp's per-source batch composition + offsets while drawing a
weighted source by class-balanced weights (focal), leaving others uniform (sc).
"""

import numpy as np
import pytest

from birdclef.data.distilled_sed_dataset import ClassBalancedMixSamp, MixSamp
from birdclef.data.xcpretrain_sampler import compute_class_weights

# focal size 100 (indices 0..99), sc size 20 (indices 100..119)
SIZES = [100, 20]
NAMES = ["focal", "sc"]
SHARES = {"focal": 0.9, "sc": 0.1}
BS = 64


def _focal_weights_skewed():
    """90 of class A + 10 of class B → class-balanced weights (B samples 9x heavier)."""
    targets = ["A"] * 90 + ["B"] * 10
    cw = compute_class_weights(targets, mode="balanced")  # A:1/90, B:1/10
    return np.array([cw[t] for t in targets], dtype=np.float64), targets


def test_is_mixsamp_subclass_and_same_composition():
    fw, _ = _focal_weights_skewed()
    cb = ClassBalancedMixSamp(
        SIZES, NAMES, SHARES, BS, nst=10, seed=0, source_weights={"focal": fw, "sc": None}
    )
    assert isinstance(cb, MixSamp)
    mx = MixSamp(SIZES, NAMES, SHARES, BS, nst=10, seed=0)
    assert cb.per_src == mx.per_src  # 58 focal + 6 sc identical composition
    # every batch is exactly BS indices
    for batch in cb:
        assert len(batch) == BS


def test_focal_draws_class_balanced():
    fw, _ = _focal_weights_skewed()
    cb = ClassBalancedMixSamp(
        SIZES, NAMES, SHARES, BS, nst=400, seed=0, source_weights={"focal": fw, "sc": None}
    )
    focal_idxs = [i for batch in cb for i in batch if i < 100]
    frac_B = np.mean([i >= 90 for i in focal_idxs])  # class-B region = indices 90..99
    # class-balanced → total A weight == total B weight → ~50% B (vs 10% uniform)
    assert frac_B > 0.35, f"class-B fraction {frac_B:.3f} not lifted toward balanced"


def test_sc_stays_uniform():
    fw, _ = _focal_weights_skewed()
    cb = ClassBalancedMixSamp(
        SIZES, NAMES, SHARES, BS, nst=400, seed=1, source_weights={"focal": fw, "sc": None}
    )
    sc_idxs = [i - 100 for batch in cb for i in batch if i >= 100]
    # uniform over 20 → each bucket ~5%; check the spread is roughly flat (max share < 0.12)
    counts = np.bincount(sc_idxs, minlength=20) / len(sc_idxs)
    assert counts.max() < 0.12


def test_all_indices_in_range():
    fw, _ = _focal_weights_skewed()
    cb = ClassBalancedMixSamp(
        SIZES, NAMES, SHARES, BS, nst=20, seed=2, source_weights={"focal": fw, "sc": None}
    )
    for batch in cb:
        assert all(0 <= i < 120 for i in batch)


def test_no_weights_matches_mixsamp_uniform():
    """source_weights all None → behaves like MixSamp (uniform), same seed → same draws."""
    cb = ClassBalancedMixSamp(SIZES, NAMES, SHARES, BS, nst=5, seed=7, source_weights=None)
    mx = MixSamp(SIZES, NAMES, SHARES, BS, nst=5, seed=7)
    assert [sorted(b) for b in cb] == [sorted(b) for b in mx]


def test_weight_length_validation():
    with pytest.raises(ValueError):
        ClassBalancedMixSamp(
            SIZES, NAMES, SHARES, BS, nst=5, seed=0, source_weights={"focal": np.ones(99)}
        )  # wrong length (size=100)
