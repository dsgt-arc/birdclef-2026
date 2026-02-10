"""Tests for pre-trained weight loading.

Exercises checkpoint loading for both conditions (16x16 and 128x1),
verifying shape matching, center-cropping, and skip logic.
"""

import pytest
import torch
from ssamba_hf import SSAMBAConfig, SSAMBAModel
from ssamba_hf.convert import convert_ssamba_weights

from src.pretrained import load_pretrained_backbone
from src.ssamba_classifier import SSAMBAClassifier
from tests.conftest import requires_pretrained


class TestWeightLoadingConditionA:
    """Condition A: 16x16 patches — patch_embed transfers, pos_embed center-cropped."""

    @requires_pretrained
    def test_load_stats(self, config_16x16, pretrained_ckpt_path, num_classes):
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        stats = load_pretrained_backbone(classifier, pretrained_ckpt_path)

        assert stats["pct"] > 95.0, f"Expected >95% params loaded, got {stats['pct']:.1f}%"
        assert len(stats["skipped"]) == 0, f"Unexpected skipped: {stats['skipped']}"
        assert any("center-cropped" in s for s in stats["special"]), (
            f"Expected pos_embed center-crop, got: {stats['special']}"
        )

    @requires_pretrained
    def test_patch_embed_copied(self, config_16x16, pretrained_ckpt_path, num_classes):
        """patch_embed.proj should be copied exactly (same shape)."""
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)

        # Save original random weights
        orig_weight = classifier.backbone.patch_embed.proj.weight.data.clone()

        load_pretrained_backbone(classifier, pretrained_ckpt_path)

        # Should have changed
        assert not torch.equal(orig_weight, classifier.backbone.patch_embed.proj.weight.data)

    @requires_pretrained
    def test_pos_embed_shape(self, config_16x16, pretrained_ckpt_path, num_classes):
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        load_pretrained_backbone(classifier, pretrained_ckpt_path)

        # 496 patches + 1 CLS = 497
        assert classifier.backbone.pos_embed.shape == (1, 497, 192)

    @requires_pretrained
    def test_mamba_layers_loaded(self, config_16x16, pretrained_ckpt_path, num_classes):
        """All 24 Mamba layer weights should be loaded from checkpoint."""
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        stats = load_pretrained_backbone(classifier, pretrained_ckpt_path)

        mamba_keys = [k for k in stats["matched"] if "layers.layers." in k]
        assert len(mamba_keys) > 0, "No Mamba layer weights loaded"


class TestWeightLoadingConditionB:
    """Condition B: 128x1 patches — only Mamba layers transfer."""

    @requires_pretrained
    def test_load_stats(self, config_128x1, pretrained_ckpt_path, num_classes):
        classifier = SSAMBAClassifier(config_128x1, num_classes=num_classes)
        stats = load_pretrained_backbone(classifier, pretrained_ckpt_path)

        # patch_embed.proj should be skipped (shape mismatch)
        assert any("patch_embed" in s for s in stats["skipped"]), (
            f"Expected patch_embed to be skipped, got skipped: {stats['skipped']}"
        )
        # pos_embed should be skipped (model needs 1001 tokens, pre-trained has 513)
        assert any("pos_embed" in s for s in stats["skipped"]), (
            f"Expected pos_embed to be skipped, got skipped: {stats['skipped']}"
        )
        # Should still load ~90%+ of params (Mamba layers)
        assert stats["pct"] > 85.0, f"Expected >85% params loaded, got {stats['pct']:.1f}%"

    @requires_pretrained
    def test_patch_embed_not_copied(self, config_128x1, pretrained_ckpt_path, num_classes):
        """patch_embed.proj should NOT be copied (shape mismatch)."""
        classifier = SSAMBAClassifier(config_128x1, num_classes=num_classes)

        # Save original random weights
        orig_weight = classifier.backbone.patch_embed.proj.weight.data.clone()

        load_pretrained_backbone(classifier, pretrained_ckpt_path)

        # Should be unchanged (random init kept)
        assert torch.equal(orig_weight, classifier.backbone.patch_embed.proj.weight.data)


class TestCheckpointStructure:
    """Verify the pre-trained checkpoint has expected structure."""

    @requires_pretrained
    def test_checkpoint_keys(self, pretrained_ckpt_path):
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=True)
        converted = convert_ssamba_weights(ckpt)

        # Should have patch_embed, pos_embed, cls_token, layers, norm_f
        assert "patch_embed.proj.weight" in converted
        assert "pos_embed" in converted
        assert "cls_token" in converted
        assert "norm_f.weight" in converted

        # Should have 24 layers (keys are layers.layers.{N}.mixer.*)
        layer_indices = {k.split(".")[2] for k in converted if k.startswith("layers.layers.")}
        assert len(layer_indices) == 24, f"Expected 24 layers, got {len(layer_indices)}: {sorted(layer_indices)}"

    @requires_pretrained
    def test_checkpoint_shapes(self, pretrained_ckpt_path):
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=True)
        converted = convert_ssamba_weights(ckpt)

        # SSAMBA Tiny: embed_dim=192, 16x16 patches, (128,1024) input
        assert converted["patch_embed.proj.weight"].shape == (192, 1, 16, 16)
        assert converted["pos_embed"].shape == (1, 513, 192)  # 512 patches + 1 CLS
        assert converted["cls_token"].shape == (1, 1, 192)
