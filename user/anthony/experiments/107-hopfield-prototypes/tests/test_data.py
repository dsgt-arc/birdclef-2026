"""Phase B data pipeline tests.

These tests verify data loading, LOSO splitting, balanced sampling, and
end-to-end training step correctness. Tests that require the exp 106
parquet cache on scratch are marked with @pytest.mark.skipif.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from birdclef_107.data import (
    EmbeddingDataset,
    LOSOFold,
    RecordingGroupedBatchSampler,
    UnsupervisedDataset,
    build_focal_sampler,
    collate_triple_batch,
    loso_folds,
)
from birdclef_107.eval import compute_auroc, pooled_auroc
from birdclef_107.family_utils import build_family_index
from birdclef_107.losses import compute_loss
from birdclef_107.model import PrototypeRetrievalModel

# Whether the exp 106 feature cache exists on scratch.
_FEATURES_EXIST = False
try:
    from pathlib import Path

    _FEATURES_EXIST = (
        Path("~/scratch/birdclef-2026/features/exp106/soundscape_labeled.parquet")
        .expanduser()
        .exists()
    )
except Exception:
    pass

needs_features = pytest.mark.skipif(
    not _FEATURES_EXIST, reason="exp 106 feature cache not available"
)


# ---------------------------------------------------------------------------
# LOSO fold tests (pure logic, no I/O)
# ---------------------------------------------------------------------------


def test_loso_folds_coverage():
    """LOSO folds cover all samples with non-overlapping splits."""
    sites = np.array(["S01", "S01", "S08", "S08", "S22", "S22", "S22"])
    folds = loso_folds(sites)

    assert len(folds) == 3  # 3 unique sites

    # Every sample appears in exactly one test set
    all_test = np.zeros(len(sites), dtype=bool)
    for fold in folds:
        assert not np.any(all_test & fold.test_mask), "overlapping test sets"
        all_test |= fold.test_mask
    assert all_test.all(), "not all samples covered"

    # Train + test = all for each fold
    for fold in folds:
        assert (fold.train_mask | fold.test_mask).all()
        assert not (fold.train_mask & fold.test_mask).any()


def test_loso_folds_sorted():
    """Folds are returned in sorted site order."""
    sites = np.array(["S22", "S08", "S01", "S22"])
    folds = loso_folds(sites)
    fold_sites = [f.left_out_site for f in folds]
    assert fold_sites == sorted(fold_sites)


def test_loso_folds_skips_empty():
    """Sites with no samples are not yielded."""
    # Only S01 and S08 have data
    sites = np.array(["S01", "S01", "S08"])
    folds = loso_folds(sites)
    fold_sites = {f.left_out_site for f in folds}
    assert fold_sites == {"S01", "S08"}


# ---------------------------------------------------------------------------
# Dataset tests (synthetic data)
# ---------------------------------------------------------------------------


def test_embedding_dataset_shapes():
    """EmbeddingDataset returns correct tensor shapes."""
    n, d, s = 20, 1536, 234
    emb = np.random.randn(n, d).astype(np.float32)
    labels = np.random.randint(0, 2, (n, s)).astype(np.int8)
    site_idx = np.arange(n, dtype=np.int64) % 3

    ds = EmbeddingDataset(emb, labels, site_idx)
    assert len(ds) == n
    item = ds[0]
    assert item["emb"].shape == (d,)
    assert item["labels"].shape == (s,)
    assert item["site_idx"].shape == ()


def test_unsupervised_dataset_shapes():
    """UnsupervisedDataset returns correct tensor shapes."""
    n, d = 20, 1536
    emb = np.random.randn(n, d).astype(np.float32)
    site_idx = np.arange(n, dtype=np.int64) % 3
    rec_idx = np.arange(n, dtype=np.int64) // 4  # 4 windows per recording

    ds = UnsupervisedDataset(emb, site_idx, recording_indices=rec_idx)
    assert len(ds) == n
    item = ds[0]
    assert item["emb"].shape == (d,)
    assert item["site_idx"].shape == ()
    assert item["recording_idx"].shape == ()


def test_unsupervised_dataset_without_recording_idx():
    """UnsupervisedDataset still works without recording_indices (back-compat)."""
    n, d = 10, 1536
    emb = np.random.randn(n, d).astype(np.float32)
    site_idx = np.arange(n, dtype=np.int64) % 3

    ds = UnsupervisedDataset(emb, site_idx)  # no recording_indices
    item = ds[0]
    assert "recording_idx" not in item


def test_recording_grouped_batch_sampler_structure():
    """Sampler emits batches of R×N indices, each batch covers R unique
    recordings, each recording contributes exactly N windows.
    """
    np.random.seed(0)
    # 20 recordings × 10 windows each = 200 total
    rec_idx = np.repeat(np.arange(20), 10).astype(np.int64)
    sampler = RecordingGroupedBatchSampler(
        rec_idx,
        recordings_per_batch=4,
        windows_per_recording=3,
        seed=0,
    )

    batches = list(sampler)
    assert len(batches) == len(sampler) == 5  # 20 recordings // 4 per batch

    for batch in batches:
        assert len(batch) == 4 * 3  # R × N = 12
        # Each batch should cover exactly 4 distinct recordings
        recs_in_batch = {int(rec_idx[i]) for i in batch}
        assert len(recs_in_batch) == 4
        # Each recording contributes exactly N windows
        from collections import Counter
        per_rec = Counter(int(rec_idx[i]) for i in batch)
        for r, count in per_rec.items():
            assert count == 3, (
                f"recording {r} appeared {count} times, expected 3"
            )


def test_recording_grouped_batch_sampler_with_replacement_for_thin_recordings():
    """If a recording has fewer than N windows, sampler falls back to
    with-replacement sampling. Ensures the batch size invariant holds.
    """
    np.random.seed(0)
    # 4 recordings × 2 windows = 8 windows; ask for N=5 per recording.
    rec_idx = np.repeat(np.arange(4), 2).astype(np.int64)
    sampler = RecordingGroupedBatchSampler(
        rec_idx,
        recordings_per_batch=2,
        windows_per_recording=5,
        seed=0,
    )

    batches = list(sampler)
    assert len(batches) >= 1
    for batch in batches:
        assert len(batch) == 2 * 5  # batch size invariant
        # Recordings can repeat windows under with-replacement
        recs = {int(rec_idx[i]) for i in batch}
        assert len(recs) == 2  # still 2 distinct recordings


def test_recording_grouped_batch_sampler_reshuffles_per_epoch():
    """Two iterations should yield different batch orderings (the
    sampler increments its internal epoch counter)."""
    rec_idx = np.repeat(np.arange(20), 5).astype(np.int64)
    sampler = RecordingGroupedBatchSampler(
        rec_idx,
        recordings_per_batch=4,
        windows_per_recording=3,
        seed=0,
    )
    ep1 = list(sampler)
    ep2 = list(sampler)
    # Same number of batches but different ordering
    assert len(ep1) == len(ep2)
    # Convert to comparable signatures
    sig1 = tuple(tuple(b) for b in ep1)
    sig2 = tuple(tuple(b) for b in ep2)
    assert sig1 != sig2


def test_recording_grouped_batch_sampler_raises_for_too_few_recordings():
    """If fewer recordings exist than R, sampler raises clearly."""
    rec_idx = np.repeat(np.arange(2), 5).astype(np.int64)
    try:
        RecordingGroupedBatchSampler(rec_idx, recordings_per_batch=4)
    except ValueError as e:
        assert "recordings available" in str(e)
    else:
        raise AssertionError("expected ValueError for insufficient recordings")


# ---------------------------------------------------------------------------
# Balanced sampler test
# ---------------------------------------------------------------------------


def test_focal_sampler_upsamples_rare():
    """Rare species should be sampled more often than their frequency."""
    np.random.seed(42)
    n = 1000
    s = 10
    labels = np.zeros((n, s), dtype=np.int8)
    # Species 0: 900 windows (common), Species 1: 10 windows (rare)
    labels[:900, 0] = 1
    labels[900:910, 1] = 1
    labels[910:, 2] = 1  # rest get species 2

    sampler = build_focal_sampler(labels)
    indices = list(sampler)
    assert len(indices) == n

    # Count how many times rare species 1 windows appear
    rare_count = sum(1 for i in indices if labels[i, 1] == 1)
    # With balanced sampling, rare species should appear much more than 10/1000 = 1%
    assert rare_count > 30, f"Rare species sampled only {rare_count} times"


# ---------------------------------------------------------------------------
# Triple batch assembly
# ---------------------------------------------------------------------------


def test_collate_triple_batch():
    """collate_triple_batch produces the dict structure compute_loss expects."""
    b = 4
    strong = {
        "emb": torch.randn(b, 1536),
        "labels": torch.zeros(b, 234),
        "site_idx": torch.zeros(b, dtype=torch.long),
    }
    focal = {
        "emb": torch.randn(b, 1536),
        "labels": torch.zeros(b, 234),
    }
    unsup = {
        "emb": torch.randn(b, 1536),
        "site_idx": torch.zeros(b, dtype=torch.long),
    }
    triple = collate_triple_batch(strong, focal, unsup)
    assert set(triple.keys()) == {"strong", "weak_focal", "unsup"}
    assert "emb" in triple["strong"]
    assert "site_idx" in triple["strong"]
    assert "labels" in triple["strong"]
    assert "emb" in triple["weak_focal"]
    assert "labels" in triple["weak_focal"]
    assert "emb" in triple["unsup"]
    assert "site_idx" in triple["unsup"]


# ---------------------------------------------------------------------------
# Training step smoke test (no I/O, synthetic data)
# ---------------------------------------------------------------------------


def test_training_step_runs():
    """A single training step completes without error on synthetic data."""
    torch.manual_seed(42)
    n_species, n_sites, b = 10, 3, 4
    d_input = 1536

    model = PrototypeRetrievalModel(
        n_species=n_species, n_sites=n_sites,
        d_input=d_input, d_hidden=64, d_proj=32,
        K_species=2, K_site=2,
    )
    # Synthetic family index: 3 families
    family_idx = torch.tensor([i % 3 for i in range(n_species)], dtype=torch.long)

    batch = {
        "strong": {
            "emb": torch.randn(b, d_input),
            "site_idx": torch.randint(0, n_sites, (b,)),
            "labels": torch.zeros(b, n_species).bernoulli_(0.1),
        },
        "weak_focal": {
            "emb": torch.randn(b, d_input),
            "labels": torch.zeros(b, n_species).bernoulli_(0.1),
        },
        "unsup": {
            "emb": torch.randn(b, d_input),
            "site_idx": torch.randint(0, n_sites, (b,)),
        },
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss, components = compute_loss(model, batch, family_idx)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    assert all(
        k in components for k in ["labeled", "site", "site_xattn", "family"]
    )


# ---------------------------------------------------------------------------
# Evaluation tests (synthetic)
# ---------------------------------------------------------------------------


def test_compute_auroc_perfect():
    """Perfect predictions give AUROC = 1.0."""
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.1]])
    assert compute_auroc(y_true, y_pred) == pytest.approx(1.0)


def test_compute_auroc_with_mask():
    """Species mask restricts evaluation to selected columns."""
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])
    mask = np.array([True, True, False])
    auroc = compute_auroc(y_true, y_pred, species_mask=mask)
    assert auroc == pytest.approx(1.0)


def test_compute_auroc_skips_single_class():
    """Species with only one class in ground truth are skipped."""
    y_true = np.array([[1, 0], [1, 0]])  # species 1 is all-zero
    y_pred = np.array([[0.9, 0.1], [0.8, 0.2]])
    # Only species 0 is evaluable (always 1) — but it's also all one class
    # Actually species 0 is all-1 → skip. Both skipped → return 0.0.
    auroc = compute_auroc(y_true, y_pred)
    assert auroc == 0.0


# ---------------------------------------------------------------------------
# Integration test with real data (requires scratch)
# ---------------------------------------------------------------------------


@needs_features
def test_load_labeled_soundscape_shapes():
    """Labeled soundscape data has expected shapes."""
    from birdclef_107.data import load_labeled_soundscape

    data = load_labeled_soundscape()
    assert data.embeddings.shape[0] == 739
    assert data.embeddings.shape[1] == 1536
    assert data.labels.shape == (739, 234)
    assert len(data.sites) == 739
    assert len(data.filenames) == 739
    assert len(data.species_list) == 234
    assert len(data.site_to_idx) > 0


@needs_features
def test_loso_folds_on_real_data():
    """LOSO folds from real data are non-overlapping and complete."""
    from birdclef_107.data import load_labeled_soundscape

    data = load_labeled_soundscape()
    folds = loso_folds(data.sites)

    # Should have multiple folds
    assert len(folds) >= 5

    # All 739 windows covered
    all_test = np.zeros(739, dtype=bool)
    for fold in folds:
        assert not np.any(all_test & fold.test_mask)
        all_test |= fold.test_mask
    assert all_test.all()
