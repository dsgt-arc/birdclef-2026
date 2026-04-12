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

    ds = UnsupervisedDataset(emb, site_idx)
    assert len(ds) == n
    item = ds[0]
    assert item["emb"].shape == (d,)
    assert item["site_idx"].shape == ()


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
    assert all(k in components for k in ["labeled", "site", "family"])


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
