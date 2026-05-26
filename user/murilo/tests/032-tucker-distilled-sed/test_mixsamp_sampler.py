"""Tests for `MixSamp` batch sampler (exp 032 task 0.7).

Validates Tucker's verbatim port from notebook cell 12:
1. Per-batch composition matches per-source shares (58 focal + 6 SC for the
   default `{"focal": 0.9, "sc": 0.1}, bs=64` — audit § 3 verification).
2. Each batch's indices land in the correct concat-dataset offset range for
   their source.
3. Sampling is with-replacement (per Tucker's `rng.integers(0, size, size=n)`).
4. Empty/zero-sized sources are skipped without raising.
5. `__len__` returns `nst` (number of batches per epoch).
6. Different seeds produce different batches; same seed reproduces.
7. Tucker's rounding-remainder distribution: shares that don't cleanly round
   to `bs` add the residual to the largest source.
8. Edge case: a source with `share=0.0` produces zero indices in the batch
   (the floor-of-1 rule applies only to non-zero shares).

Reference:
- src/birdclef/data/distilled_sed_dataset.py — MixSamp
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§ 3)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 12)
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.distilled_sed_dataset import MixSamp

# ============================================================================
# Tucker's default: bs=64, shares={"focal":0.9, "sc":0.1} → 58 + 6
# ============================================================================


def test_tucker_default_composition_58_6():
    """Audit § 3: bs=64 with shares={focal:0.9, sc:0.1} yields 58 + 6 per batch."""
    sampler = MixSamp(
        sizes=[1000, 100],
        names=["focal", "sc"],
        shares={"focal": 0.9, "sc": 0.1},
        bs=64,
        nst=10,
        seed=0,
    )
    assert sampler.per_src == [58, 6]
    assert sum(sampler.per_src) == 64

    for batch in sampler:
        assert len(batch) == 64
        focal_count = sum(1 for i in batch if i < 1000)
        sc_count = sum(1 for i in batch if i >= 1000)
        assert focal_count == 58
        assert sc_count == 6


def test_offsets_translate_correctly():
    """Indices from source `i` must live in [offsets[i], offsets[i]+sizes[i])."""
    sampler = MixSamp(
        sizes=[100, 50, 20],
        names=["a", "b", "c"],
        shares={"a": 0.5, "b": 0.3, "c": 0.2},
        bs=10,
        nst=5,
        seed=42,
    )
    # offsets = [0, 100, 150]
    assert sampler.offsets == [0, 100, 150]
    for batch in sampler:
        # All indices must fit into one of the three source ranges
        for idx in batch:
            in_a = 0 <= idx < 100
            in_b = 100 <= idx < 150
            in_c = 150 <= idx < 170
            assert in_a or in_b or in_c, f"index {idx} falls outside all source ranges"


def test_batch_size_exact_after_rounding_remainder():
    """When `bs * share` doesn't round cleanly to `bs`, remainder is added to
    the largest source — final batch size still exactly `bs`."""
    # 0.33 × 30 = 9.9 → rounds to 10 each → sum=30. No remainder here.
    sampler = MixSamp(
        sizes=[100, 100, 100],
        names=["a", "b", "c"],
        shares={"a": 0.33, "b": 0.33, "c": 0.33},
        bs=30,
        nst=3,
        seed=0,
    )
    assert sum(sampler.per_src) == 30

    # 0.7 × 10 = 7, 0.2 × 10 = 2, 0.1 × 10 = 1 → sum=10. Exact.
    sampler2 = MixSamp(
        sizes=[100, 100, 100],
        names=["a", "b", "c"],
        shares={"a": 0.7, "b": 0.2, "c": 0.1},
        bs=10,
        nst=3,
        seed=0,
    )
    assert sampler2.per_src == [7, 2, 1]

    # 0.95 × 11 = 10.45 → rounds to 10; 0.05 × 11 = 0.55 → rounds to 1 (floor=1).
    # sum=11. Exact.
    sampler3 = MixSamp(
        sizes=[100, 100],
        names=["a", "b"],
        shares={"a": 0.95, "b": 0.05},
        bs=11,
        nst=3,
        seed=0,
    )
    assert sum(sampler3.per_src) == 11


def test_zero_share_explicit_opts_out():
    """When a source has share=0.0 EXACTLY, it gets zero samples (overrides
    the `max(1, ...)` floor that applies to small-but-positive shares)."""
    sampler = MixSamp(
        sizes=[100, 100],
        names=["a", "b"],
        shares={"a": 1.0, "b": 0.0},
        bs=10,
        nst=5,
        seed=0,
    )
    assert sampler.per_src == [10, 0]
    for batch in sampler:
        # All indices must be from source `a` (offset 0, size 100)
        for idx in batch:
            assert 0 <= idx < 100, f"got idx {idx} but source b is opted out"


def test_small_nonzero_share_gets_floor_of_one():
    """A small-but-nonzero share rounds to floor=1, not 0. Tucker's behavior
    (a source with share=0.05 and bs=10 would round to 0.5 → 1)."""
    sampler = MixSamp(
        sizes=[100, 100],
        names=["a", "b"],
        shares={"a": 0.95, "b": 0.05},
        bs=10,
        nst=3,
        seed=0,
    )
    # 0.95 × 10 = 9.5 → round to 10 (because of max(1, ...) and remainder
    # distribution); 0.05 × 10 = 0.5 → max(1, 0) = 1 → sum=11 → remainder
    # -1 distributed to argmax (source a). Final: [9, 1].
    # Whatever the exact split, source b must have count >= 1.
    assert sampler.per_src[1] >= 1, f"got per_src={sampler.per_src}"


# ============================================================================
# __len__ and stochasticity
# ============================================================================


def test_len_equals_nst():
    nst = 42
    sampler = MixSamp(
        sizes=[100, 50],
        names=["a", "b"],
        shares={"a": 0.5, "b": 0.5},
        bs=8,
        nst=nst,
        seed=0,
    )
    assert len(sampler) == nst
    n_iterations = sum(1 for _ in sampler)
    assert n_iterations == nst


def test_same_seed_reproduces_same_batches():
    s1 = MixSamp([100, 50], ["a", "b"], {"a": 0.5, "b": 0.5}, bs=8, nst=10, seed=42)
    s2 = MixSamp([100, 50], ["a", "b"], {"a": 0.5, "b": 0.5}, bs=8, nst=10, seed=42)
    batches_1 = list(s1)
    batches_2 = list(s2)
    assert batches_1 == batches_2


def test_different_seed_produces_different_batches():
    """Per the docstring's note about per-epoch seed shifts: changing seed
    must change the batch composition (Tucker uses `seed=42 + ep`)."""
    s1 = MixSamp([100, 50], ["a", "b"], {"a": 0.5, "b": 0.5}, bs=8, nst=10, seed=42)
    s2 = MixSamp([100, 50], ["a", "b"], {"a": 0.5, "b": 0.5}, bs=8, nst=10, seed=43)
    batches_1 = list(s1)
    batches_2 = list(s2)
    assert batches_1 != batches_2


def test_within_batch_shuffle_applied():
    """The combined batch is shuffled (rng.shuffle), so consecutive source
    blocks are interleaved, not contiguous. Empirically verified by checking
    that indices from one source are not all in the first N positions."""
    sampler = MixSamp(
        sizes=[1000, 100],
        names=["focal", "sc"],
        shares={"focal": 0.9, "sc": 0.1},
        bs=64,
        nst=20,
        seed=42,
    )
    # If unshuffled: batch[0:58] would all be focal (idx < 1000); batch[58:64]
    # would all be sc (idx >= 1000). Shuffling should mix these.
    found_interleaved = False
    for batch in sampler:
        # Look for an sc-index in the first 58 positions
        if any(idx >= 1000 for idx in batch[:58]):
            found_interleaved = True
            break
    assert found_interleaved, "MixSamp batches don't appear to be shuffled"


# ============================================================================
# With-replacement sampling
# ============================================================================


def test_sampling_with_replacement():
    """`rng.integers(0, size, size=n)` is uniform with replacement; small
    pool + large batch must produce duplicates."""
    # 10 samples per source, take 50 → must have duplicates
    sampler = MixSamp(
        sizes=[10],
        names=["a"],
        shares={"a": 1.0},
        bs=50,
        nst=10,
        seed=0,
    )
    for batch in sampler:
        unique = len(set(batch))
        assert unique < len(
            batch
        ), f"expected duplicates with bs=50 and source size=10, got {unique} unique"


def test_empirical_rate_matches_share():
    """Over many batches, the fraction of each source's indices in the
    combined batch pool should match the per-source share."""
    sampler = MixSamp(
        sizes=[1000, 100],
        names=["focal", "sc"],
        shares={"focal": 0.9, "sc": 0.1},
        bs=64,
        nst=100,
        seed=0,
    )
    counts = Counter()
    for batch in sampler:
        for idx in batch:
            if idx < 1000:
                counts["focal"] += 1
            else:
                counts["sc"] += 1
    total = counts["focal"] + counts["sc"]
    focal_frac = counts["focal"] / total
    sc_frac = counts["sc"] / total
    # 58/64 = 0.90625; 6/64 = 0.09375 — exact, not just expected.
    assert abs(focal_frac - 58 / 64) < 1e-9, f"focal_frac={focal_frac:.5f}"
    assert abs(sc_frac - 6 / 64) < 1e-9, f"sc_frac={sc_frac:.5f}"


# ============================================================================
# Edge cases
# ============================================================================


def test_zero_sized_source_skipped():
    """A source with size=0 is skipped — useful for empty SS pool when fold_k
    holds out all of them."""
    sampler = MixSamp(
        sizes=[100, 0],
        names=["a", "b"],
        shares={"a": 0.9, "b": 0.1},
        bs=10,
        nst=5,
        seed=0,
    )
    # Per-source counts: a=9, b=1. But b has size 0, so __iter__ skips it.
    for batch in sampler:
        # Batch will have only 9 indices (from source a) — not bs, since
        # source b can't contribute. That's Tucker's behavior.
        for idx in batch:
            assert 0 <= idx < 100  # only source a indices


def test_invalid_inputs_raise():
    """Mismatched sizes/names lengths raise ValueError."""
    with pytest.raises(ValueError, match="sizes.*names"):
        MixSamp(sizes=[100, 50], names=["a"], shares={"a": 1.0}, bs=10, nst=5)
    with pytest.raises(ValueError, match="bs must be > 0"):
        MixSamp(sizes=[100], names=["a"], shares={"a": 1.0}, bs=0, nst=5)
    with pytest.raises(ValueError, match="nst must be >= 0"):
        MixSamp(sizes=[100], names=["a"], shares={"a": 1.0}, bs=10, nst=-1)


def test_iter_dataloader_integration():
    """End-to-end with `DataLoader` + `ConcatDataset`: each batch yields
    `bs` indices that can be successfully retrieved from a wrapped concat."""
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    class _DummyDS(Dataset):
        def __init__(self, n, tag):
            self.n = n
            self.tag = tag

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return (torch.tensor([i]), self.tag)

    focal_ds = _DummyDS(1000, "focal")
    sc_ds = _DummyDS(100, "sc")
    concat = ConcatDataset([focal_ds, sc_ds])
    sampler = MixSamp(
        sizes=[1000, 100],
        names=["focal", "sc"],
        shares={"focal": 0.9, "sc": 0.1},
        bs=64,
        nst=5,
        seed=0,
    )

    def _collate(batch):
        idxs = torch.cat([b[0] for b in batch])
        tags = [b[1] for b in batch]
        return idxs, tags

    loader = DataLoader(concat, batch_sampler=sampler, num_workers=0, collate_fn=_collate)
    n_batches = 0
    for _idxs, tags in loader:
        n_batches += 1
        assert len(tags) == 64
        # Composition: 58 focal + 6 sc per batch (from earlier test)
        assert tags.count("focal") == 58
        assert tags.count("sc") == 6
    assert n_batches == 5
