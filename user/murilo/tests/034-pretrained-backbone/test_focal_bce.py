"""Phase B unit tests for FocalLossBCE (B.6).

Covers:
- finite loss + finite gradient on a synthetic batch
- bce_weight=0 → exactly torchvision.sigmoid_focal_loss
- focal_weight=0 → exactly BCEWithLogitsLoss
- default params match Sydorskyi's verified config (α=0.25, γ=2.0, 1.0/1.0, mean)
"""

import torch
import torch.nn as nn
import torchvision

from birdclef.losses.focal_bce import FocalLossBCE


def test_finite_loss_and_grad():
    loss_fn = FocalLossBCE()
    logits = torch.randn(4, 234, requires_grad=True)
    targets = (torch.rand(4, 234) > 0.9).float()
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_matches_focal_when_bce_weight_zero():
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    targets = (torch.rand(4, 10) > 0.5).float()
    fn = FocalLossBCE(bce_weight=0.0, focal_weight=1.0)
    ref = torchvision.ops.focal_loss.sigmoid_focal_loss(
        logits, targets, alpha=0.25, gamma=2.0, reduction="mean"
    )
    torch.testing.assert_close(fn(logits, targets), ref)


def test_matches_bce_when_focal_weight_zero():
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    targets = (torch.rand(4, 10) > 0.5).float()
    fn = FocalLossBCE(bce_weight=1.0, focal_weight=0.0)
    ref = nn.BCEWithLogitsLoss(reduction="mean")(logits, targets)
    torch.testing.assert_close(fn(logits, targets), ref)


def test_default_params_match_verified_config():
    fn = FocalLossBCE()
    assert fn.alpha == 0.25
    assert fn.gamma == 2.0
    assert fn.bce_weight == 1.0
    assert fn.focal_weight == 1.0
    assert fn.reduction == "mean"


def test_composition_is_weighted_sum():
    torch.manual_seed(1)
    logits = torch.randn(3, 8)
    targets = (torch.rand(3, 8) > 0.5).float()
    fn = FocalLossBCE(bce_weight=1.0, focal_weight=1.0)
    bce = nn.BCEWithLogitsLoss(reduction="mean")(logits, targets)
    focal = torchvision.ops.focal_loss.sigmoid_focal_loss(
        logits, targets, alpha=0.25, gamma=2.0, reduction="mean"
    )
    torch.testing.assert_close(fn(logits, targets), bce + focal)
