"""Phase B unit tests for the Sydorskyi 2025 SED-attention head port.

Covers:
- Constructor defaults match Sydorskyi: hidden_chans=512, p=0.5, num_class=234 (our 2026 default)
- Dropout positions: 0.25 before linear (p/2), 0.5 after ReLU (p)
- Activation = ReLU
- Output dict has clipwise_logits_long (LOGIT space), clipwise_pred_long (PROB space), framewise_logits_long
- Output shapes: (B, num_class) for clipwise, (B, time, num_class) for framewise
- Outputs are finite; logit-space output unbounded; prob-space output in [0, 1]
- train_period != infer_period raises (we explicitly simplified to remove that path)
"""

import pytest
import torch
import torch.nn as nn

from birdclef.heads.yoink_sed_head import YoinkAttHead


def _backbone_features_like(B: int, C: int, H_freq: int = 4, W_time: int = 10) -> torch.Tensor:
    """Build a dummy backbone feature map of the shape produced by EffNetV2-S
    or NFNet-L0 features_only mode (verified Phase A.1.4 + Phase B smoke test)."""
    return torch.randn(B, C, H_freq, W_time)


def test_constructor_defaults_match_sydorskyi():
    head = YoinkAttHead(in_chans=256)
    # Hidden dim 512 is the Sydorskyi-verified value
    assert head.dense_layers[1].in_features == 256
    assert head.dense_layers[1].out_features == 512
    # Num classes default = 234 (BirdCLEF 2026)
    assert head.attention.out_channels == 234
    assert head.fix_scale.out_channels == 234


def test_dropout_positions_and_rates():
    """Master p=0.5 → dropout(p/2=0.25) before linear, dropout(p=0.5) after ReLU."""
    head = YoinkAttHead(in_chans=256, p=0.5)
    dropouts = [m for m in head.dense_layers if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 2
    assert dropouts[0].p == pytest.approx(0.25), f"first dropout = {dropouts[0].p}, expected 0.25"
    assert dropouts[1].p == pytest.approx(0.5), f"second dropout = {dropouts[1].p}, expected 0.5"


def test_activation_is_relu():
    head = YoinkAttHead(in_chans=256)
    activations = [m for m in head.dense_layers if isinstance(m, nn.ReLU)]
    assert len(activations) == 1, "expected exactly one ReLU activation in dense_layers"


def test_forward_output_dict_keys_and_shapes():
    """Returns dict with {clipwise_pred_long, clipwise_logits_long, framewise_logits_long}."""
    head = YoinkAttHead(in_chans=256, num_class=234)
    feat = _backbone_features_like(B=2, C=256)  # mimics EffNetV2-S features_only output
    out = head(feat)

    assert set(out.keys()) == {
        "clipwise_pred_long",
        "clipwise_logits_long",
        "framewise_logits_long",
    }
    assert out["clipwise_pred_long"].shape == (2, 234)
    assert out["clipwise_logits_long"].shape == (2, 234)
    assert out["framewise_logits_long"].shape == (2, 10, 234)


def test_forward_outputs_finite():
    head = YoinkAttHead(in_chans=2304, num_class=234)  # mimics NFNet-L0 features_only output
    feat = _backbone_features_like(B=2, C=2304)
    out = head(feat)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} produced non-finite values"


def test_clipwise_pred_is_probability_space():
    """clipwise_pred_long = sum(sigmoid(feat_v) * softmax_t(time_att), dim=t) — must be in [0, 1]."""
    head = YoinkAttHead(in_chans=256)
    head.eval()
    feat = _backbone_features_like(B=2, C=256) * 5.0  # large dynamic range
    out = head(feat)
    pred = out["clipwise_pred_long"]
    assert (pred >= 0).all() and (
        pred <= 1
    ).all(), f"clipwise_pred_long out of [0, 1]: min={pred.min().item()}, max={pred.max().item()}"


def test_clipwise_logits_unbounded():
    """clipwise_logits_long = sum(feat_v * softmax_t(time_att), dim=t) — should NOT be saturated in [0, 1]."""
    head = YoinkAttHead(in_chans=256)
    head.eval()
    feat = _backbone_features_like(B=4, C=256) * 5.0
    out = head(feat)
    logits = out["clipwise_logits_long"]
    # If aggregation is in logit space (Sydorskyi style), some values should be < 0 or > 1
    # (the soft-attention-weighted sum of raw logits can be anywhere on the real line).
    assert (logits < 0).any() or (
        logits > 1
    ).any(), "clipwise_logits_long appears bounded — suggests prob-space aggregation bug"


def test_simplification_assertion_train_period_must_equal_infer_period():
    """Our simplified head requires train_period == infer_period. Confirm explicit error."""
    with pytest.raises(AssertionError, match="train_period == infer_period"):
        YoinkAttHead(in_chans=256, train_period=15.0, infer_period=5.0)


def test_gradient_flow_end_to_end():
    """A backward pass through the head must propagate gradients to all learnable params."""
    head = YoinkAttHead(in_chans=256, num_class=234, p=0.5)
    head.train()
    feat = _backbone_features_like(B=2, C=256)
    feat.requires_grad_(True)
    out = head(feat)
    loss = out["clipwise_logits_long"].abs().mean()
    loss.backward()
    # Check GeMFreq.p got gradient, attention conv got gradient, fix_scale conv got gradient
    assert head.pooling.p.grad is not None and head.pooling.p.grad.abs().sum() > 0
    assert head.attention.weight.grad is not None and head.attention.weight.grad.abs().sum() > 0
    assert head.fix_scale.weight.grad is not None and head.fix_scale.weight.grad.abs().sum() > 0
