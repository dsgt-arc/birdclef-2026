"""Tests for TuckerMacroAUC / TuckerMacroAP (skip-empty-class metrics).

Locks in the bug-fix for the torchmetrics 1.9 behavior where 0-positive
classes contribute 0.0 to the macro average (drags val/cmap and val/auroc
on exp 032 from 0.89 down to 0.11).

Reference: `scripts/diag_val_auroc.py`, audit notes 2026-05-23.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics.classification import MultilabelAUROC

from birdclef.metrics import TuckerMacroAP, TuckerMacroAUC

# ----------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sparse_multilabel_data(rng):
    """168 samples × 234 classes, sparse positives (~1.5 / sample) — matches
    exp 032's val fold statistics."""
    num_samples, num_classes = 168, 234
    p_pos = 1.5 / num_classes
    y_true = (rng.random((num_samples, num_classes)) < p_pos).astype(np.float32)
    # mildly informative predictor (matches the diagnostic script's setup)
    logits = y_true * 2.0 + rng.normal(0, 1.0, y_true.shape).astype(np.float32)
    y_pred = 1.0 / (1.0 + np.exp(-logits))
    return y_true, y_pred


# ----------------------------------------------------------------------
# Sklearn equivalence (the contract we're matching)
# ----------------------------------------------------------------------


def tucker_reference_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """Verbatim port of compute_macro_auc from Tucker's notebook lines 369-384."""
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def test_macro_auc_matches_tucker_reference(sparse_multilabel_data):
    """TuckerMacroAUC.compute() == sklearn-based reference port."""
    y_true, y_pred = sparse_multilabel_data
    metric = TuckerMacroAUC(num_labels=234)
    metric.update(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    auc_ours = float(metric.compute())
    auc_ref, _ = tucker_reference_auc(y_true, y_pred)
    assert math.isclose(auc_ours, auc_ref, rel_tol=1e-6), (auc_ours, auc_ref)


def test_macro_ap_matches_sklearn(sparse_multilabel_data):
    """TuckerMacroAP.compute() == sklearn-based reference port (mean AP over evaluable)."""
    y_true, y_pred = sparse_multilabel_data
    metric = TuckerMacroAP(num_labels=234)
    metric.update(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    ap_ours = float(metric.compute())
    aps = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        aps.append(average_precision_score(col, y_pred[:, c]))
    ap_ref = float(np.mean(aps))
    assert math.isclose(ap_ours, ap_ref, rel_tol=1e-6), (ap_ours, ap_ref)


# ----------------------------------------------------------------------
# The actual fix — verify we DON'T match the broken torchmetrics behavior
# ----------------------------------------------------------------------


def test_tucker_macro_auc_is_higher_than_torchmetrics_on_sparse(sparse_multilabel_data):
    """Demonstrates the bug fix: TuckerMacroAUC > MultilabelAUROC when many
    classes have 0 positives, because torchmetrics counts them as 0.0."""
    y_true, y_pred = sparse_multilabel_data
    n_zero = int((y_true.sum(axis=0) == 0).sum())
    assert n_zero > 0, "fixture must have some 0-positive classes"

    ours = TuckerMacroAUC(num_labels=234)
    ours.update(torch.from_numpy(y_pred), torch.from_numpy(y_true))

    theirs = MultilabelAUROC(num_labels=234, average="macro")
    theirs.update(torch.from_numpy(y_pred), torch.from_numpy(y_true).int())

    assert float(ours.compute()) > float(theirs.compute()) + 0.1


# ----------------------------------------------------------------------
# Skip-empty-class semantics
# ----------------------------------------------------------------------


def test_skips_zero_positive_classes():
    """Classes with no positive samples must not contribute to the macro mean."""
    # 3 classes: class 0 has positives, classes 1 and 2 are all-zero.
    y_true = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
    y_pred = torch.tensor(
        [[0.9, 0.5, 0.5], [0.1, 0.5, 0.5], [0.8, 0.5, 0.5], [0.2, 0.5, 0.5]],
        dtype=torch.float32,
    )
    metric = TuckerMacroAUC(num_labels=3)
    metric.update(y_pred, y_true)
    # Only class 0 is evaluable → AUC=1.0 (positives ranked above negatives)
    assert float(metric.compute()) == pytest.approx(1.0)
    assert metric.n_evaluable() == 1


def test_skips_all_positive_classes():
    """Classes where every sample is positive must also be skipped."""
    # Class 0 has mixed, class 1 is all-positive (also AUC-undefined).
    y_true = torch.tensor([[1, 1], [0, 1], [1, 1], [0, 1]], dtype=torch.float32)
    y_pred = torch.tensor(
        [[0.9, 0.5], [0.1, 0.5], [0.8, 0.5], [0.2, 0.5]],
        dtype=torch.float32,
    )
    metric = TuckerMacroAUC(num_labels=2)
    metric.update(y_pred, y_true)
    assert float(metric.compute()) == pytest.approx(1.0)
    assert metric.n_evaluable() == 1


def test_returns_nan_when_no_evaluable_classes():
    """No-positive-anywhere case → NaN, not zero."""
    y_true = torch.zeros(4, 5)
    y_pred = torch.rand(4, 5)
    metric = TuckerMacroAUC(num_labels=5)
    metric.update(y_pred, y_true)
    result = float(metric.compute())
    assert math.isnan(result)
    assert metric.n_evaluable() == 0


# ----------------------------------------------------------------------
# Lightning lifecycle: update / compute / reset / multi-batch accumulation
# ----------------------------------------------------------------------


def test_accumulates_across_multiple_updates():
    """Per-batch update calls must accumulate into one global metric at compute()."""
    metric = TuckerMacroAUC(num_labels=3)
    # Batch 1: perfect ordering for class 0
    metric.update(torch.tensor([[0.9, 0.5, 0.5]]), torch.tensor([[1.0, 0.0, 0.0]]))
    # Batch 2: perfect ordering for class 0
    metric.update(torch.tensor([[0.1, 0.5, 0.5]]), torch.tensor([[0.0, 0.0, 0.0]]))
    # After 2 batches: class 0 has 1 pos + 1 neg, perfect ordering → AUC=1.0
    assert float(metric.compute()) == pytest.approx(1.0)
    assert metric.n_evaluable() == 1


def test_reset_clears_state():
    """Lightning calls reset() between epochs."""
    metric = TuckerMacroAUC(num_labels=2)
    metric.update(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    _ = metric.compute()
    metric.reset()
    assert len(metric.preds) == 0
    assert len(metric.target) == 0
    # After reset + no updates, compute() returns NaN
    assert math.isnan(float(metric.compute()))


# ----------------------------------------------------------------------
# Input-validation guards
# ----------------------------------------------------------------------


def test_rejects_wrong_num_labels():
    metric = TuckerMacroAUC(num_labels=10)
    bad_preds = torch.rand(4, 5)
    bad_target = torch.zeros(4, 5)
    with pytest.raises(ValueError, match="Expected preds shape"):
        # num_labels=10 in metric but preds is (4, 5)
        metric.update(bad_preds, bad_target)


def test_rejects_shape_mismatch():
    metric = TuckerMacroAUC(num_labels=5)
    preds = torch.rand(4, 5)
    target = torch.zeros(3, 5)  # different batch size
    with pytest.raises(ValueError, match="shape mismatch"):
        metric.update(preds, target)


# ----------------------------------------------------------------------
# Determinism + thread safety
# ----------------------------------------------------------------------


def test_deterministic_on_same_input(sparse_multilabel_data):
    y_true, y_pred = sparse_multilabel_data
    results = []
    for _ in range(3):
        metric = TuckerMacroAUC(num_labels=234)
        metric.update(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        results.append(float(metric.compute()))
    assert all(r == results[0] for r in results)
