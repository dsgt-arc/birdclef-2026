"""Tests for SEDLightningModule — CPU unit tests."""

import torch
import torch.nn as nn

from src.lightning_module import SEDLightningModule


class _MockSEDModel(nn.Module):
    """Minimal mock that mimics SSAMBASEDClassifier interface."""

    def __init__(self, num_classes, num_tokens=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        # Named to match freeze patterns
        self.sed_head = nn.Linear(32, num_classes)
        self.mel_spec = nn.Linear(1, 1)  # dummy
        self.backbone = nn.ModuleDict({
            "patch_embed": nn.Linear(1, 1),
            "layer0": nn.Linear(1, 1),
        })

    def forward(self, waveforms, padding_mask=None, spec_augment=None):
        B = waveforms.shape[0]
        clip_pred = torch.sigmoid(torch.randn(B, self.num_classes))
        sed_logits = torch.randn(B, self.num_tokens, self.num_classes)
        return clip_pred, sed_logits


class TestSEDLightningModule:
    def test_training_step_returns_loss(self):
        num_classes = 5
        model = _MockSEDModel(num_classes)
        module = SEDLightningModule(model=model, num_classes=num_classes)

        batch = (
            torch.randn(4, 32000),                    # waveforms
            torch.zeros(4, num_classes),               # clip_labels
            torch.zeros(4, 100, num_classes),          # token_labels
            torch.ones(4, 100, dtype=torch.bool),      # padding_mask
        )

        loss = module.training_step(batch, 0)
        assert loss.isfinite()
        assert loss.dim() == 0

    def test_validation_step_returns_loss(self):
        num_classes = 5
        model = _MockSEDModel(num_classes)
        module = SEDLightningModule(model=model, num_classes=num_classes)

        batch = (
            torch.randn(4, 32000),
            torch.zeros(4, num_classes),
            torch.zeros(4, 100, num_classes),
            torch.ones(4, 100, dtype=torch.bool),
        )

        loss = module.validation_step(batch, 0)
        assert loss.isfinite()

    def test_configure_optimizers(self):
        num_classes = 5
        model = _MockSEDModel(num_classes)
        module = SEDLightningModule(model=model, num_classes=num_classes)

        opt_dict = module.configure_optimizers()
        assert "optimizer" in opt_dict
        assert "lr_scheduler" in opt_dict

    def test_alpha_zero_no_sed_loss(self):
        """With alpha=0, total loss should equal clip loss."""
        num_classes = 5
        model = _MockSEDModel(num_classes)
        module = SEDLightningModule(model=model, num_classes=num_classes, alpha=0.0)

        batch = (
            torch.randn(4, 32000),
            torch.zeros(4, num_classes),
            torch.zeros(4, 100, num_classes),
            torch.ones(4, 100, dtype=torch.bool),
        )

        loss = module.training_step(batch, 0)
        assert loss.isfinite()
