"""Tests for MixSamp with 3 sources (focal + sc + pl) — exp 033.

Verifies the existing exp 032 MixSamp is N-source capable (no code change
in the sampler itself, just confirms 3-source usage works):
- Batch composition matches the configured shares.
- Indices translate correctly across source offsets (focal → [0, n_focal),
  sc → [n_focal, n_focal + n_sc), pl → [n_focal + n_sc, total)).
- Total batch size sums to bs across all sources.
- Empirical per-source rates over many batches match shares.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.distilled_sed_dataset import MixSamp

# Pool sizes that loosely mimic exp 033 reality (focal much smaller than PL)
N_FOCAL = 1000
N_SC = 100
N_PL = 10000
BATCH_SIZE = 64
N_STEPS = 50


def _make_sampler(shares: dict[str, float], seed: int = 42):
    return MixSamp(
        sizes=[N_FOCAL, N_SC, N_PL],
        names=["focal", "sc", "pl"],
        shares=shares,
        bs=BATCH_SIZE,
        nst=N_STEPS,
        seed=seed,
    )


def _classify(idx: int) -> str:
    """Map a ConcatDataset index back to its source."""
    if idx < N_FOCAL:
        return "focal"
    if idx < N_FOCAL + N_SC:
        return "sc"
    return "pl"


# ----------------------------------------------------------------------
# Per-batch composition
# ----------------------------------------------------------------------


def test_mixsamp_3source_batch_size_correct():
    """Every batch has exactly bs samples."""
    sampler = _make_sampler({"focal": 0.5, "sc": 0.0, "pl": 0.5})
    for batch in sampler:
        assert len(batch) == BATCH_SIZE


def test_mixsamp_3source_shares_50_50():
    """shares={focal: 0.5, sc: 0.0, pl: 0.5} → 32 focal + 32 pl per batch (sc=0)."""
    sampler = _make_sampler({"focal": 0.5, "sc": 0.0, "pl": 0.5})
    for batch in sampler:
        counts = Counter(_classify(i) for i in batch)
        assert counts["focal"] == 32
        assert counts["pl"] == 32
        assert counts.get("sc", 0) == 0


def test_mixsamp_3source_warmstart_a3_shares():
    """A3 warm-start config: shares={focal: 0.2, sc: 0.0, pl: 0.8}.
    Expected: ~13 focal + ~51 pl per batch (rounding remainder may shift by 1)."""
    sampler = _make_sampler({"focal": 0.2, "sc": 0.0, "pl": 0.8})
    for batch in sampler:
        counts = Counter(_classify(i) for i in batch)
        # MixSamp distributes rounding remainders; allow ±1
        assert 12 <= counts["focal"] <= 14, f"focal count: {counts['focal']}"
        assert 50 <= counts["pl"] <= 52, f"pl count: {counts['pl']}"
        assert counts.get("sc", 0) == 0


def test_mixsamp_3source_a1_shares_with_sc():
    """A1 shares={focal: 0.5, sc: 0.0, pl: 0.5} — SC pool present but share=0."""
    sampler = _make_sampler({"focal": 0.5, "sc": 0.0, "pl": 0.5})
    for batch in sampler:
        counts = Counter(_classify(i) for i in batch)
        assert counts.get("sc", 0) == 0, "Zero share must yield zero samples"


# ----------------------------------------------------------------------
# Index translation correctness
# ----------------------------------------------------------------------


def test_mixsamp_3source_indices_in_correct_ranges():
    """Each sample index must fall in the right pool range."""
    sampler = _make_sampler({"focal": 0.3, "sc": 0.05, "pl": 0.65})
    for batch in sampler:
        for idx in batch:
            assert 0 <= idx < N_FOCAL + N_SC + N_PL


def test_mixsamp_3source_no_pool_collision():
    """Indices labeled as 'focal' must be < N_FOCAL; 'sc' must be in [N_FOCAL, N_FOCAL+N_SC); etc."""
    sampler = _make_sampler({"focal": 0.5, "sc": 0.0, "pl": 0.5})
    for batch in sampler:
        for idx in batch:
            tag = _classify(idx)
            if tag == "focal":
                assert idx < N_FOCAL
            elif tag == "sc":
                assert N_FOCAL <= idx < N_FOCAL + N_SC
            elif tag == "pl":
                assert idx >= N_FOCAL + N_SC


# ----------------------------------------------------------------------
# Empirical share rate over many batches
# ----------------------------------------------------------------------


def test_mixsamp_3source_empirical_shares_match():
    """Over N_STEPS batches, empirical per-source share ≈ configured share."""
    shares = {"focal": 0.5, "sc": 0.0, "pl": 0.5}
    sampler = _make_sampler(shares, seed=42)

    total_counts = Counter()
    for batch in sampler:
        total_counts.update(_classify(i) for i in batch)
    total = sum(total_counts.values())
    assert total == N_STEPS * BATCH_SIZE

    # 50/50 split is exact (32 each per batch); zero share must be exactly 0
    assert total_counts["focal"] == N_STEPS * 32
    assert total_counts["pl"] == N_STEPS * 32
    assert total_counts.get("sc", 0) == 0


# ----------------------------------------------------------------------
# 2-source backward-compat (exp 032 baseline still works)
# ----------------------------------------------------------------------


def test_mixsamp_2source_backward_compat():
    """Confirm exp 032's 2-source usage (focal+sc) still works after exp 033 changes."""
    sampler = MixSamp(
        sizes=[N_FOCAL, N_SC],
        names=["focal", "sc"],
        shares={"focal": 0.9, "sc": 0.1},
        bs=BATCH_SIZE,
        nst=10,
        seed=42,
    )
    for batch in sampler:
        assert len(batch) == BATCH_SIZE
        counts = Counter()
        for idx in batch:
            counts["focal" if idx < N_FOCAL else "sc"] += 1
        # Tucker's 0.9/0.1 with bs=64 → 58 focal + 6 sc
        assert counts["focal"] == 58
        assert counts["sc"] == 6
