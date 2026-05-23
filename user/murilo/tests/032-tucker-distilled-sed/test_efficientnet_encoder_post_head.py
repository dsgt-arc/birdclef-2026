"""Tests for `EfficientNetEncoder(use_post_head=True)` — exp 032 task 0.2.

Verifies the new Tucker-mode flag on the encoder:
1. Default (features_only) mode unchanged — EffNet-B0 still gives 320-d output
2. use_post_head=True mode gives 1280-d output (matches Tucker's notebook)
3. End-to-end wiring: encoder → AttHeadLogitSpace returns expected shapes
4. tf_efficientnet_b0.ns_jft_in1k is a valid timm model name in both modes
5. Forward output is finite on random input
6. HGNet (non-EffNet) still works in default mode (back-compat receipt)

Pretrained=False throughout to keep tests fast and offline-runnable. Pretrained
load is verified manually outside the test suite.

Reference:
- user/murilo/src/birdclef/encoders/efficientnet.py — use_post_head flag
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md (backbone setup)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 10)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder
from birdclef.heads.sed_head import AttHeadLogitSpace

NUM_CLASSES = 234
BATCH = 2
N_MELS = 256
TIME_FRAMES = 313  # 5s @ 32kHz with hop=512


# ============================================================================
# Default mode (features_only) — back-compat receipt
# ============================================================================


def test_default_mode_unchanged_effnet_b0():
    """EffNet-B0 default mode still gives 320-d (timm features_only last stage)."""
    enc = EfficientNetEncoder("tf_efficientnet_b0.ns_jft_in1k", pretrained=False)
    assert enc.embed_dim == 320
    assert enc.use_post_head is False


def test_default_mode_unchanged_hgnet():
    """HGNetV2-B0 default mode gives 1024-d (the v5j backbone)."""
    enc = EfficientNetEncoder("hgnetv2_b0.ssld_stage2_ft_in1k", pretrained=False)
    assert enc.embed_dim == 1024
    assert enc.use_post_head is False


def test_default_mode_forward_returns_list_last():
    """forward_features should still extract the last stage when use_post_head=False."""
    enc = EfficientNetEncoder("tf_efficientnet_b0.ns_jft_in1k", pretrained=False)
    mel = torch.randn(BATCH, 1, N_MELS, TIME_FRAMES)
    feat = enc.forward_features(mel)
    # EffNet-B0 last stage: (B, 320, H/32, W/32) = (2, 320, 8, 10)
    assert feat.shape == (BATCH, 320, 8, 10)


# ============================================================================
# Tucker mode (use_post_head=True)
# ============================================================================


def test_tucker_mode_embed_dim_1280():
    """EffNet-B0 NS-JFT in Tucker mode gives 1280-d (post-conv_head)."""
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        drop_path_rate=0.1,
        use_post_head=True,
    )
    assert enc.embed_dim == 1280
    assert enc.use_post_head is True


def test_tucker_mode_forward_returns_single_tensor():
    """forward_features in Tucker mode returns a single (B, 1280, F', T') tensor."""
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        use_post_head=True,
    )
    mel = torch.randn(BATCH, 1, N_MELS, TIME_FRAMES)
    feat = enc.forward_features(mel)
    # 5s @ 32kHz, hop=512, n_mels=256 → mel (2, 1, 256, 313) → backbone /32 → (8, 10)
    assert feat.shape == (BATCH, 1280, 8, 10)
    assert torch.isfinite(feat).all()


def test_tucker_mode_output_is_4d():
    """Output must remain 4D (no global pool applied)."""
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    feat = enc.forward_features(torch.randn(BATCH, 1, N_MELS, TIME_FRAMES))
    assert feat.dim() == 4


def test_tucker_mode_pooled_forward_works():
    """The forward() method (global-pool aggregation) should still work."""
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    pooled = enc(torch.randn(BATCH, 1, N_MELS, TIME_FRAMES))
    assert pooled.shape == (BATCH, 1280)


# ============================================================================
# End-to-end: encoder + AttHeadLogitSpace
# ============================================================================


def test_end_to_end_tucker_encoder_to_att_head_logit_space():
    """Full Phase 0.1 + 0.2 wiring: Tucker encoder → AttHeadLogitSpace.

    Confirms output shapes are what training_step / validation_step expect.
    """
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        drop_path_rate=0.1,
        use_post_head=True,
    )
    head = AttHeadLogitSpace(
        in_channels=enc.embed_dim,  # 1280
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        dropout=0.5,
        gem_p=3.0,
    )
    mel = torch.randn(BATCH, 1, N_MELS, TIME_FRAMES)
    feat = enc.forward_features(mel)
    clip_logits, frame_logits = head(feat)
    assert clip_logits.shape == (BATCH, NUM_CLASSES)
    # Time axis preserved by AttHead (no temporal pool in encoder)
    assert frame_logits.shape == (BATCH, NUM_CLASSES, 10)


def test_end_to_end_gradient_flow():
    """Gradient flows through the encoder + head from a BCE loss on clipwise."""
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        use_post_head=True,
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=NUM_CLASSES)
    mel = torch.randn(BATCH, 1, N_MELS, TIME_FRAMES, requires_grad=True)

    feat = enc.forward_features(mel)
    clip_logits, _ = head(feat)
    target = torch.zeros(BATCH, NUM_CLASSES)
    target[:, 0] = 1.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(clip_logits, target)
    loss.backward()

    # mel got gradient (encoder is in the graph)
    assert mel.grad is not None
    assert mel.grad.abs().sum().item() > 0
    # encoder conv_stem also got gradient
    assert enc.backbone.conv_stem.weight.grad is not None


# ============================================================================
# Sanity: spatial dims unchanged across mode toggle
# ============================================================================


def test_spatial_dims_match_across_modes():
    """Toggling use_post_head should change ONLY the channel dim, not spatial.
    Both modes apply the same conv backbone with same strides — the only
    difference is whether conv_head (channel projection) runs."""
    enc_default = EfficientNetEncoder("tf_efficientnet_b0.ns_jft_in1k", pretrained=False)
    enc_tucker = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    mel = torch.randn(BATCH, 1, N_MELS, TIME_FRAMES)
    feat_default = enc_default.forward_features(mel)
    feat_tucker = enc_tucker.forward_features(mel)
    # Same batch, same spatial; different channels
    assert feat_default.shape[0] == feat_tucker.shape[0]
    assert feat_default.shape[2:] == feat_tucker.shape[2:]
    assert feat_default.shape[1] == 320
    assert feat_tucker.shape[1] == 1280


def test_drop_path_rate_propagates_in_tucker_mode():
    """drop_path_rate should be honored in Tucker mode (Tucker uses 0.1)."""
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        drop_path_rate=0.1,
        use_post_head=True,
    )
    # Smoke test: model instantiates without error; forward runs
    feat = enc.forward_features(torch.randn(BATCH, 1, N_MELS, TIME_FRAMES))
    assert feat.shape == (BATCH, 1280, 8, 10)
