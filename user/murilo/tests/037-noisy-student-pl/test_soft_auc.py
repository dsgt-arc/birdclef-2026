"""Exp 037 — SoftAUCLoss tests.

Pins the ranking semantics: perfect ranking → ~0, reversed → large, monotone in
ranking quality, reduces to standard AUC surrogate on binary labels, handles
no-variation batches, and is differentiable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.losses import SoftAUCLoss, soft_auc_loss


def test_perfect_ranking_near_zero():
    # 1 class, scores perfectly co-ordered with (soft) labels → low loss
    logits = torch.tensor([[5.0], [3.0], [1.0], [-2.0]])
    targets = torch.tensor([[1.0], [0.7], [0.3], [0.0]])
    loss = soft_auc_loss(logits, targets)
    assert loss.item() < 0.05


def test_reversed_ranking_large():
    logits = torch.tensor([[-5.0], [-3.0], [1.0], [5.0]])  # anti-correlated with labels
    targets = torch.tensor([[1.0], [0.7], [0.3], [0.0]])
    loss = soft_auc_loss(logits, targets)
    assert loss.item() > 2.0


def test_monotone_in_ranking_quality():
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    good = soft_auc_loss(torch.tensor([[4.0], [-4.0], [3.0], [-3.0]]), targets)
    bad = soft_auc_loss(torch.tensor([[0.5], [0.1], [0.2], [0.3]]), targets)
    worse = soft_auc_loss(torch.tensor([[-4.0], [4.0], [-3.0], [3.0]]), targets)
    assert good.item() < bad.item() < worse.item()


def test_binary_reduces_to_auc_surrogate():
    # all positives outscore all negatives → AUC=1 → near-zero surrogate
    logits = torch.tensor([[10.0], [9.0], [-9.0], [-10.0]])
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    assert soft_auc_loss(logits, targets).item() < 1e-3


def test_no_label_variation_returns_zero():
    # every sample shares the same label for the class → no pair weight → 0
    logits = torch.randn(8, 5)
    targets = torch.full((8, 5), 0.4)
    loss = soft_auc_loss(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_multiclass_macro_and_range():
    torch.manual_seed(0)
    logits = torch.randn(16, 234)
    targets = (torch.rand(16, 234) > 0.9).float()  # sparse multi-hot
    loss = soft_auc_loss(logits, targets)
    assert loss.item() >= 0.0 and torch.isfinite(loss)


def test_differentiable():
    logits = torch.randn(8, 4, requires_grad=True)
    targets = (torch.rand(8, 4) > 0.5).float()
    loss = soft_auc_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_module_matches_function():
    logits = torch.randn(8, 4)
    targets = (torch.rand(8, 4) > 0.5).float()
    assert torch.allclose(SoftAUCLoss()(logits, targets), soft_auc_loss(logits, targets))
