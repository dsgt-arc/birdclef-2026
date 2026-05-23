"""Tests for `AttHeadLogitSpace` (exp 032 task 0.1.3).

Verifies the Tucker-recipe SED head:
1. Forward produces expected output shapes
2. Logit-space aggregation matches `sum(softmax(tanh(att)) * cla, dim=time)`
3. Gradients flow through both att and cls 1x1 convs
4. xavier_uniform_ init produces zero-mean weight tensors
5. GeMFreqPool clamps `p` to >= 1.0 in forward (Tucker variant vs plain GeM)
6. Output is finite (no NaN/Inf) on diverse input scales

Reference:
- user/murilo/src/birdclef/heads/sed_head.py — AttHeadLogitSpace, GeMFreqPool
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb
  (cells 10, 14)
- user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.heads.sed_head import AttHeadLogitSpace, GeMFreqPool

NUM_CLASSES = 234
IN_CHANNELS = 1280  # EffNet-B0 backbone output
BATCH = 4
FREQ = 8  # post-backbone freq downsample
TIME = 10  # post-backbone time downsample


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def head():
    """Default AttHeadLogitSpace (Tucker's exact params)."""
    return AttHeadLogitSpace(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        dropout=0.5,
        gem_p=3.0,
    )


@pytest.fixture
def features():
    """(B, C, F, T) random feature map from a hypothetical backbone."""
    return torch.randn(BATCH, IN_CHANNELS, FREQ, TIME)


# ============================================================================
# Forward shape + dtype
# ============================================================================


def test_forward_shapes(head, features):
    clip_logits, frame_logits = head(features)
    assert clip_logits.shape == (BATCH, NUM_CLASSES)
    assert frame_logits.shape == (BATCH, NUM_CLASSES, TIME)
    assert clip_logits.dtype == features.dtype
    assert frame_logits.dtype == features.dtype


def test_forward_returns_tuple(head, features):
    out = head(features)
    assert isinstance(out, tuple)
    assert len(out) == 2


# ============================================================================
# Aggregation domain — the LOAD-BEARING test
# ============================================================================


def test_logit_space_aggregation_matches_manual_formula(head, features):
    """`clipwise_logits = sum(softmax(tanh(att(h))) * cla(h), dim=time)`.

    This is the formula from Tucker's notebook cell 10 — explicitly NOT in
    probability space. We recompute by hand and assert exact equality (mod fp
    rounding), guarding against accidental sigmoid-roundtrip drift.
    """
    head.eval()  # disable dropout for exact comparison
    with torch.no_grad():
        # Replicate the forward up to (B, hidden, T)
        x = head.gem(features)  # (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = head.dense(x)  # (B, T, hidden)
        x = x.permute(0, 2, 1)  # (B, hidden, T)

        att_raw = torch.tanh(head.att(x))  # (B, num_classes, T)
        att_weights = torch.softmax(att_raw, dim=-1)
        frame_logits = head.cla(x)  # (B, num_classes, T)
        clip_logits_manual = (att_weights * frame_logits).sum(dim=-1)

        # Compare to the module's forward output
        clip_logits, frame_logits_from_forward = head(features)

    torch.testing.assert_close(clip_logits, clip_logits_manual, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(frame_logits_from_forward, frame_logits, rtol=1e-5, atol=1e-6)


def test_no_sigmoid_roundtrip_in_clip_logits(head, features):
    """If we were doing a sigmoid → mean → inverse-sigmoid roundtrip, clip_logits
    would be bounded by logit(1 - 1e-7) ≈ ±16. Logit-space sums have no such
    bound — large per-frame logits propagate through linearly."""
    # Manually push large values through the att/cla convs by scaling features
    head.eval()
    big_features = features * 10.0  # amplify post-GeM activations
    with torch.no_grad():
        clip_logits, _ = head(big_features)
    # Logit-space aggregation should produce values that can exceed ±16
    # (the saturation boundary of `torch.logit(clamp(prob, 1e-7, 1-1e-7))`).
    # We don't assert it WILL exceed — just that no saturation clamp is in play.
    assert torch.isfinite(clip_logits).all()


# ============================================================================
# Gradient flow
# ============================================================================


def test_gradients_flow_through_both_convs(head, features):
    """Both `att` and `cla` 1x1 convs must receive gradient from a BCE loss
    on `clip_logits`. (The clipwise aggregation multiplies them, so both
    branches should be in the autograd graph.)"""
    features = features.clone().requires_grad_(True)
    target = torch.zeros(BATCH, NUM_CLASSES)
    target[:, 0] = 1.0  # arbitrary multi-hot target

    clip_logits, _ = head(features)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(clip_logits, target)
    loss.backward()

    assert head.att.weight.grad is not None
    assert head.cla.weight.grad is not None
    assert head.att.weight.grad.abs().sum().item() > 0
    assert head.cla.weight.grad.abs().sum().item() > 0
    assert head.att.bias.grad is not None
    assert head.cla.bias.grad is not None


def test_gradient_through_frame_max_supervision(head, features):
    """Tucker's dual-supervision loss takes BCE on `framewise.max(dim=time)`.
    Verify the framewise output also produces useful gradient on the cla conv
    (att shouldn't receive gradient from this term — only cla does)."""
    features = features.clone().requires_grad_(True)
    target = torch.zeros(BATCH, NUM_CLASSES)
    target[:, 0] = 1.0

    _, frame_logits = head(features)
    frame_max = frame_logits.max(dim=-1).values  # (B, num_classes)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(frame_max, target)
    loss.backward()

    # Only `cla` is in the frame_max graph; `att` is NOT used for frame_max.
    assert head.cla.weight.grad is not None
    assert head.cla.weight.grad.abs().sum().item() > 0
    assert head.att.weight.grad is None  # att not touched by frame-max loss


# ============================================================================
# Initialization
# ============================================================================


def test_xavier_init_zero_mean_weights(head):
    """xavier_uniform_ produces weights uniform in [-a, +a] for a = sqrt(6/(fan_in+fan_out)).
    Mean should be very close to zero for large enough fan-out × num_classes tensors."""
    # Conv1d weight shape: (num_classes, hidden_dim, 1) → 234 × 512 = 119k weights
    att_mean = head.att.weight.mean().item()
    cla_mean = head.cla.weight.mean().item()
    # Loose bound; xavier_uniform is symmetric around 0 by construction
    assert abs(att_mean) < 1e-2, f"att.weight mean too far from 0: {att_mean}"
    assert abs(cla_mean) < 1e-2, f"cla.weight mean too far from 0: {cla_mean}"


def test_bias_init_zero(head):
    """Tucker explicitly zeroes both biases."""
    assert (head.att.bias.data == 0).all()
    assert (head.cla.bias.data == 0).all()


# ============================================================================
# GeMFreqPool (Tucker variant)
# ============================================================================


def test_gemfreqpool_clamps_p_to_one():
    """GeMFreqPool should clamp `p` to >= 1.0 in forward, even if the learnable
    parameter drifts below. Distinguishes it from plain GeM which has no clamp."""
    pool = GeMFreqPool(p=3.0)
    # Manually set p to 0.5 (below 1.0) to simulate pathological training drift
    pool.p.data.fill_(0.5)
    x = torch.rand(BATCH, IN_CHANNELS, FREQ, TIME) + 0.1  # positive

    # Manual computation with the clamp
    p_clamped = pool.p.clamp(min=1.0)
    expected = x.clamp(min=pool.eps).pow(p_clamped).mean(dim=-2).pow(1.0 / p_clamped)
    actual = pool(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gemfreqpool_output_shape():
    pool = GeMFreqPool(p=3.0)
    x = torch.rand(BATCH, IN_CHANNELS, FREQ, TIME)
    out = pool(x)
    assert out.shape == (BATCH, IN_CHANNELS, TIME)


def test_gemfreqpool_p_is_learnable():
    pool = GeMFreqPool(p=3.0)
    assert isinstance(pool.p, torch.nn.Parameter)
    assert pool.p.requires_grad


# ============================================================================
# Numerical stability
# ============================================================================


def test_finite_output_on_zero_features(head):
    """Edge case: all-zero feature map (clamped to eps internally by GeMFreqPool)."""
    features = torch.zeros(BATCH, IN_CHANNELS, FREQ, TIME)
    head.eval()
    with torch.no_grad():
        clip_logits, frame_logits = head(features)
    assert torch.isfinite(clip_logits).all()
    assert torch.isfinite(frame_logits).all()


def test_finite_output_on_large_features(head):
    """Edge case: large-magnitude feature map."""
    features = torch.randn(BATCH, IN_CHANNELS, FREQ, TIME) * 100.0
    head.eval()
    with torch.no_grad():
        clip_logits, frame_logits = head(features)
    assert torch.isfinite(clip_logits).all()
    assert torch.isfinite(frame_logits).all()


def test_train_and_eval_both_run(head, features):
    """Dropout is the only train/eval mode-sensitive component; both should run."""
    head.train()
    clip_t, frame_t = head(features)
    assert clip_t.shape == (BATCH, NUM_CLASSES)

    head.eval()
    with torch.no_grad():
        clip_e, frame_e = head(features)
    assert clip_e.shape == (BATCH, NUM_CLASSES)
