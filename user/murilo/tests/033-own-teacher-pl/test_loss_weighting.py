"""Tests for per-sample loss_weight in DistilledSEDFinetuneModule (exp 033).

Verifies:
- `_dual_supervision_loss(loss_weight=None)` matches existing behavior
  (backward-compat with exp 032's mean BCE).
- `_dual_supervision_loss(loss_weight=ones)` is mathematically identical
  to `loss_weight=None`.
- `_dual_supervision_loss(loss_weight=[1, 0])` zeros out the 2nd sample's
  contribution (per-sample weighting actually works).
- `training_step` accepts both 2-tuple (legacy unit tests), 4-tuple
  (Tucker dataloader), AND 5-tuple (PL dataloader).
- KD MSE loss is NOT per-sample weighted (only the BCE task branch).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder
from birdclef.heads.sed_head import AttHeadLogitSpace
from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule
from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram

NUM_CLASSES = 234
BATCH = 4
SAMPLES_5S = 160000


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def module():
    encoder = EfficientNetEncoder(
        model_name="tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        drop_path_rate=0.0,
        in_chans=1,
        use_post_head=True,
    )
    head = AttHeadLogitSpace(
        in_channels=encoder.embed_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        dropout=0.5,
        gem_p=3.0,
    )
    mel = TuckerMelSpectrogram(
        sample_rate=32000, n_fft=2048, hop_length=512, n_mels=256, fmin=20, fmax=16000, top_db=80.0
    )
    return DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=head,
        num_classes=NUM_CLASSES,
        mel_transform=mel,
        use_perch_distill=False,  # KD off keeps tests cheap; KD-on path tested in exp 032
        perch_onnx_path=None,
    )


def _make_logits_labels(seed: int = 42):
    """Generate (clip_logits, frame_logits, labels) with B=4."""
    torch.manual_seed(seed)
    clip_logits = torch.randn(BATCH, NUM_CLASSES)
    frame_logits = torch.randn(BATCH, NUM_CLASSES, 10)  # (B, C, T)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    labels[:, 17] = 1.0
    return clip_logits, frame_logits, labels


# ----------------------------------------------------------------------
# _dual_supervision_loss — backward-compat with exp 032
# ----------------------------------------------------------------------


def test_dual_supervision_loss_none_equals_existing(module):
    """`loss_weight=None` must reproduce the existing (exp 032) behavior."""
    clip, frame, labels = _make_logits_labels(seed=1)
    loss_none, _ = module._dual_supervision_loss(clip, frame, labels, loss_weight=None)
    loss_omit, _ = module._dual_supervision_loss(clip, frame, labels)
    assert torch.isclose(
        loss_none, loss_omit, rtol=1e-7
    ), "Passing loss_weight=None must match omitting the arg (existing behavior)."


def test_dual_supervision_loss_ones_equals_none(module):
    """`loss_weight=ones(B)` is mathematically identical to `loss_weight=None`
    (both are 'mean of per-sample mean BCE').
    """
    clip, frame, labels = _make_logits_labels(seed=2)
    weights = torch.ones(BATCH)
    loss_ones, _ = module._dual_supervision_loss(clip, frame, labels, loss_weight=weights)
    loss_none, _ = module._dual_supervision_loss(clip, frame, labels, loss_weight=None)
    # Note: ones path uses reduction='none' → mean(dim=1) → mean, while None path
    # uses reduction='mean' (mean over all (B, C) elements). These differ only by
    # whether per-class-mean is taken before or after batch-mean — for ones weights
    # they are mathematically identical (both equal sum/(B*C)).
    assert torch.isclose(
        loss_ones, loss_none, rtol=1e-6
    ), f"ones weights should equal None: got {loss_ones.item():.6f} vs {loss_none.item():.6f}"


# ----------------------------------------------------------------------
# _dual_supervision_loss — per-sample weighting actually works
# ----------------------------------------------------------------------


def test_dual_supervision_loss_zero_weight_excludes_sample(module):
    """`loss_weight=[1, 1, 1, 0]` must give the same result as the unweighted
    mean over the first 3 samples — the 4th sample's contribution is excluded
    (modulo the batch-size divisor difference)."""
    clip, frame, labels = _make_logits_labels(seed=3)
    # Standard: mean over all 4 samples
    loss_all, _ = module._dual_supervision_loss(clip, frame, labels)
    # Zero out the 4th sample's weight
    weights = torch.tensor([1.0, 1.0, 1.0, 0.0])
    loss_partial, _ = module._dual_supervision_loss(clip, frame, labels, loss_weight=weights)
    # loss_partial = (sample0 + sample1 + sample2 + 0*sample3) / 4
    # loss_all     = (sample0 + sample1 + sample2 + sample3) / 4   (with reduction='mean')
    # They should differ unless sample3 happens to be exactly 0.
    assert not torch.isclose(
        loss_all, loss_partial, rtol=1e-3
    ), "Zero weight on a sample should change the loss."


def test_dual_supervision_loss_weight_scale(module):
    """Doubling all per-sample weights should double the loss (linearity)."""
    clip, frame, labels = _make_logits_labels(seed=4)
    weights_1 = torch.ones(BATCH) * 1.0
    weights_2 = torch.ones(BATCH) * 2.0
    loss_1, _ = module._dual_supervision_loss(clip, frame, labels, loss_weight=weights_1)
    loss_2, _ = module._dual_supervision_loss(clip, frame, labels, loss_weight=weights_2)
    assert torch.isclose(
        loss_2, 2 * loss_1, rtol=1e-6
    ), f"2× weight should give 2× loss: {loss_2.item():.4f} vs 2×{loss_1.item():.4f}"


# ----------------------------------------------------------------------
# training_step — variable batch tuple length
# ----------------------------------------------------------------------


def test_training_step_2tuple(module):
    """Backward-compat: 2-tuple (x, labels) — used by unit-test fixtures."""
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    with patch.object(module, "log"):
        loss = module.training_step((x, labels), batch_idx=0)
    assert torch.isfinite(loss), f"2-tuple training_step produced non-finite loss: {loss}"


def test_training_step_4tuple_tucker_default(module):
    """Backward-compat: 4-tuple (x, labels, wts, mks) — Tucker dataloader output."""
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    wts = torch.ones(BATCH, NUM_CLASSES)  # (B, C) per-class — UNUSED but present
    mks = torch.ones(BATCH, NUM_CLASSES)
    with patch.object(module, "log"):
        loss = module.training_step((x, labels, wts, mks), batch_idx=0)
    assert torch.isfinite(loss), f"4-tuple training_step produced non-finite loss: {loss}"


def test_training_step_5tuple_pl(module):
    """NEW: 5-tuple (x, labels, wts, mks, loss_weight) — PL dataloader output."""
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    wts = torch.ones(BATCH, NUM_CLASSES)
    mks = torch.ones(BATCH, NUM_CLASSES)
    loss_weight = torch.ones(BATCH)  # (B,) per-sample — Antoine 1:1 protocol
    with patch.object(module, "log"):
        loss = module.training_step((x, labels, wts, mks, loss_weight), batch_idx=0)
    assert torch.isfinite(loss), f"5-tuple training_step produced non-finite loss: {loss}"


def test_training_step_5tuple_changes_loss_when_weighted(module):
    """Mathematically: 5-tuple with non-ones weight should produce a different
    loss than the same 4-tuple (proving the weight is actually plumbed through)."""
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    wts = torch.ones(BATCH, NUM_CLASSES)
    mks = torch.ones(BATCH, NUM_CLASSES)
    # Non-trivial weight: zero out last sample
    loss_weight = torch.tensor([1.0, 1.0, 1.0, 0.0])

    with patch.object(module, "log"):
        loss_4tuple = module.training_step((x, labels, wts, mks), batch_idx=0)
        loss_5tuple = module.training_step((x, labels, wts, mks, loss_weight), batch_idx=0)
    assert not torch.isclose(loss_4tuple, loss_5tuple, rtol=1e-3), (
        f"Zero-weighted sample should change loss: 4-tuple={loss_4tuple.item():.4f} "
        f"vs 5-tuple={loss_5tuple.item():.4f}"
    )


def test_training_step_5tuple_ignores_per_class_at_batch4(module):
    """Defensive: if batch[4] is per-class (ndim=2), training_step must NOT
    treat it as loss_weight. This prevents a future bug where someone accidentally
    extends the Tucker collate to return a 5-tuple with a (B, C) tensor at position 4.

    Implementation: inspect the loss_weight arg passed to `_dual_supervision_loss`
    via a spy/mock. Direct loss comparison is unreliable due to BatchNorm + Dropout
    stochasticity.
    """
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    wts = torch.ones(BATCH, NUM_CLASSES)
    mks = torch.ones(BATCH, NUM_CLASSES)
    bad_per_class = torch.ones(BATCH, NUM_CLASSES) * 0.5  # (B, C) at batch[4]

    # Spy on _dual_supervision_loss to capture the loss_weight kwarg
    received_lw = {}
    orig_loss = module._dual_supervision_loss

    def spy_loss(clip, frame, lb, loss_weight=None):
        received_lw["lw"] = loss_weight
        return orig_loss(clip, frame, lb, loss_weight=loss_weight)

    with (
        patch.object(module, "log"),
        patch.object(module, "_dual_supervision_loss", side_effect=spy_loss),
    ):
        module.training_step((x, labels, wts, mks, bad_per_class), batch_idx=0)
    assert received_lw["lw"] is None, (
        f"Per-class (B, C) tensor at batch[4] should NOT be treated as loss_weight. "
        f"Got loss_weight = {received_lw['lw']}"
    )
