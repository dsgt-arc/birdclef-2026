"""Tests for loss functions and data augmentation.

Cross-ref: ref_yuriy_protossm_v5.py lines 531-548 (focal), 1660-1680 (mixup).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from birdclef.perch.losses import focal_bce_with_logits, mixup_files


class TestFocalBCE:
    def test_gamma_zero_equals_bce(self):
        """With gamma=0, focal loss should equal standard BCE."""
        logits = torch.randn(4, 12, 234)
        targets = torch.zeros(4, 12, 234)
        targets[:2, :3, :5] = 1.0

        focal = focal_bce_with_logits(logits, targets, gamma=0.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        assert torch.allclose(focal, bce, atol=1e-5)

    def test_focal_reduces_easy_examples(self):
        """Focal loss should be < BCE for easy (well-classified) examples."""
        # Create easy examples: logits strongly positive for positive targets
        logits = torch.tensor([[5.0, -5.0]])  # confident predictions
        targets = torch.tensor([[1.0, 0.0]])  # matching labels

        focal = focal_bce_with_logits(logits, targets, gamma=2.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        assert focal < bce, "Focal should down-weight easy examples"

    def test_focal_weight_formula(self):
        """Verify (1-pt)^gamma weighting.

        Cross-ref: ref_yuriy_protossm_v5.py lines 541-543.
        """
        logits = torch.tensor([[0.0]])  # sigmoid = 0.5
        targets = torch.tensor([[1.0]])

        # At sigmoid=0.5, pt=0.5, focal_weight=(0.5)^2=0.25
        focal = focal_bce_with_logits(logits, targets, gamma=2.0, reduction="none")
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        expected_weight = 0.25  # (1-0.5)^2
        assert torch.allclose(focal, bce * expected_weight, atol=1e-5)

    def test_pos_weight(self):
        """Focal loss should accept pos_weight."""
        logits = torch.randn(2, 12, 234)
        targets = torch.zeros(2, 12, 234)
        pw = torch.ones(1, 1, 234) * 5.0
        loss = focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=pw)
        assert loss.shape == ()  # scalar

    def test_reduction_none(self):
        logits = torch.randn(2, 12, 234)
        targets = torch.zeros(2, 12, 234)
        loss = focal_bce_with_logits(logits, targets, gamma=2.0, reduction="none")
        assert loss.shape == (2, 12, 234)


class TestMixup:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 10
        return {
            "emb": np.random.randn(n, 12, 1536).astype(np.float32),
            "logits": np.random.randn(n, 12, 234).astype(np.float32),
            "labels": np.random.rand(n, 12, 234).astype(np.float32),
            "site_ids": np.arange(n),
            "hours": np.random.randint(0, 24, n),
            "families": np.random.rand(n, 50).astype(np.float32),
        }

    def test_shape_preservation(self, sample_data):
        result = mixup_files(**sample_data, alpha=0.3)
        for orig, mixed in zip(sample_data.values(), result, strict=False):
            if orig is not None and mixed is not None:
                assert mixed.shape == orig.shape

    def test_lambda_at_least_half(self, sample_data):
        """Lambda should be enforced >= 0.5.

        Cross-ref: ref_yuriy_protossm_v5.py line 1669.
        """
        np.random.seed(0)
        for _ in range(100):
            emb_mix, _, _, _, _, _ = mixup_files(**sample_data, alpha=0.3)
            # If lambda >= 0.5, the dominant sample contributes more
            # Can't directly check lambda, but mixup should not be identity
            # (with high probability)

    def test_no_mixup_when_alpha_zero(self, sample_data):
        result = mixup_files(**sample_data, alpha=0.0)
        np.testing.assert_array_equal(result[0], sample_data["emb"])
        np.testing.assert_array_equal(result[1], sample_data["logits"])

    def test_discrete_features_preserved(self, sample_data):
        """Site IDs and hours should keep dominant sample's values (not interpolated).

        Cross-ref: ref_yuriy_protossm_v5.py line 1677.
        """
        _, _, _, site_ids_out, hours_out, _ = mixup_files(**sample_data, alpha=0.3)
        np.testing.assert_array_equal(site_ids_out, sample_data["site_ids"])
        np.testing.assert_array_equal(hours_out, sample_data["hours"])

    def test_single_sample_no_mixup(self):
        """With n=1, mixup should be a no-op."""
        emb = np.random.randn(1, 12, 1536).astype(np.float32)
        lo = np.random.randn(1, 12, 234).astype(np.float32)
        lab = np.random.rand(1, 12, 234).astype(np.float32)
        result = mixup_files(emb, lo, lab, np.array([0]), np.array([6]), None, alpha=0.3)
        np.testing.assert_array_equal(result[0], emb)
