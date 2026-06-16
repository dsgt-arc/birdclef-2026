"""Exp 038 — module loss-path tests: focalbce + label_smoothing are additive & gated; the bce
path stays bit-identical to exp 032/033/037. Exercises DistilledSEDFinetuneModule._dual_supervision_loss
directly with nn.Identity stubs (no backbone/mel/network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule  # noqa: E402


def _module(loss_type="bce", label_smoothing=0.0):
    return DistilledSEDFinetuneModule(
        encoder=nn.Identity(),
        sed_head=nn.Identity(),
        num_classes=4,
        mel_transform=nn.Identity(),
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        clipwise_weight=0.5,
        framewise_weight=0.5,
    )


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    clip = torch.randn(8, 4, generator=g)
    frame = torch.randn(8, 4, 6, generator=g)
    labels = (torch.rand(8, 4, generator=g) > 0.5).float()
    return clip, frame, labels


def test_bce_path_bit_identical_to_canonical():
    """Regression guard: adding focalbce/label_smoothing branches must not change bce."""
    m = _module("bce", 0.0)
    clip, frame, labels = _inputs()
    loss, frame_max = m._dual_supervision_loss(clip, frame, labels)

    fm = frame.max(dim=-1).values
    expected = 0.5 * F.binary_cross_entropy_with_logits(
        clip, labels.float()
    ) + 0.5 * F.binary_cross_entropy_with_logits(fm, labels.float())
    assert torch.allclose(loss, expected, atol=1e-12, rtol=0)
    assert torch.equal(frame_max, fm)
    assert m.focal_loss is None  # not instantiated for bce


def test_label_smoothing_applies_to_targets_only():
    clip, frame, labels = _inputs()
    l0, _ = _module("bce", 0.0)._dual_supervision_loss(clip, frame, labels)
    l1, _ = _module("bce", 0.05)._dual_supervision_loss(clip, frame, labels)
    assert not torch.allclose(l0, l1)  # smoothing changes the loss

    tgt = labels.float() * (1.0 - 0.05)
    fm = frame.max(dim=-1).values
    expected = 0.5 * F.binary_cross_entropy_with_logits(
        clip, tgt
    ) + 0.5 * F.binary_cross_entropy_with_logits(fm, tgt)
    assert torch.allclose(l1, expected, atol=1e-6)


def test_focalbce_path_runs_and_is_finite():
    m = _module("focalbce", 0.05)
    assert m.focal_loss is not None
    clip, frame, labels = _inputs()
    loss, _ = m._dual_supervision_loss(clip, frame, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_invalid_loss_type_raises():
    with pytest.raises(ValueError, match="loss_type"):
        _module("nonsense")


def test_invalid_label_smoothing_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        _module("bce", 1.5)
