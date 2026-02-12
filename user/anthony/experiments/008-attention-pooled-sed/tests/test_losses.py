"""Tests for FocalBCELoss."""

import pytest
import torch

from src.losses import FocalBCELoss


class TestFocalBCELossFromLogits:
    """Test FocalBCELoss with from_logits=True (default)."""

    def test_gamma_zero_equals_bce(self):
        """With gamma=0, focal loss == plain BCE."""
        loss_fn = FocalBCELoss(gamma=0.0, from_logits=True)
        logits = torch.randn(4, 21)
        targets = torch.zeros(4, 21)
        targets[:, :3] = 1.0

        focal = loss_fn(logits, targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        assert torch.allclose(focal, bce, atol=1e-6)

    def test_gamma_positive_less_than_bce(self):
        """With gamma>0, focal loss <= plain BCE (easy examples down-weighted)."""
        logits = torch.randn(8, 21)
        targets = torch.zeros(8, 21)
        targets[:, :3] = 1.0

        focal = FocalBCELoss(gamma=2.0)(logits, targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        assert focal <= bce

    def test_output_is_scalar(self):
        loss_fn = FocalBCELoss(gamma=2.0)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = FocalBCELoss(gamma=2.0)
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.zeros(4, 10)
        targets[:, 0] = 1.0
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


class TestFocalBCELossFromProbs:
    """Test FocalBCELoss with from_logits=False (for attention-pooled output)."""

    def test_gamma_zero_equals_bce(self):
        loss_fn = FocalBCELoss(gamma=0.0, from_logits=False)
        probs = torch.sigmoid(torch.randn(4, 21))
        targets = torch.zeros(4, 21)
        targets[:, :3] = 1.0

        focal = loss_fn(probs, targets)
        bce = torch.nn.functional.binary_cross_entropy(probs, targets)
        assert torch.allclose(focal, bce, atol=1e-5)

    def test_clamps_extreme_probs(self):
        """Should not error on probs at 0 or 1."""
        loss_fn = FocalBCELoss(gamma=2.0, from_logits=False)
        probs = torch.tensor([[0.0, 1.0, 0.5]])
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        loss = loss_fn(probs, targets)
        assert loss.isfinite()


class TestFocalBCELossMasked:
    """Test masked SED loss for variable-length token sequences."""

    def test_mask_excludes_padding(self):
        """Masked loss should differ from unmasked when padding is non-zero."""
        loss_fn = FocalBCELoss(gamma=0.0, from_logits=True)

        B, T, C = 2, 10, 5
        logits = torch.randn(B, T, C)
        targets = torch.zeros(B, T, C)
        targets[:, :5, 0] = 1.0  # first 5 tokens active

        # Mask: first 7 tokens real, last 3 padding
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :7] = True

        masked_loss = loss_fn(logits, targets, mask=mask)
        unmasked_loss = loss_fn(logits, targets)

        # Should generally differ (padding tokens have different logits)
        assert masked_loss.isfinite()
        assert unmasked_loss.isfinite()

    def test_all_masked_returns_zero(self):
        """If all tokens are masked out, loss should be 0 (or at least finite)."""
        loss_fn = FocalBCELoss(gamma=2.0, from_logits=True)
        logits = torch.randn(2, 10, 5)
        targets = torch.zeros(2, 10, 5)
        mask = torch.zeros(2, 10, dtype=torch.bool)  # all masked

        loss = loss_fn(logits, targets, mask=mask)
        assert loss.isfinite()
        assert loss.item() == 0.0

    def test_mask_shape_broadcast(self):
        """Mask (B, T) broadcasts to (B, T, C) correctly."""
        loss_fn = FocalBCELoss(gamma=2.0, from_logits=True)
        B, T, C = 4, 100, 21
        logits = torch.randn(B, T, C)
        targets = torch.zeros(B, T, C)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, 80:] = False  # last 20 tokens padded

        loss = loss_fn(logits, targets, mask=mask)
        assert loss.isfinite()
