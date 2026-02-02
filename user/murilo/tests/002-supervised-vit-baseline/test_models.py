"""Test ViTAudio model for Experiment 002."""

import pytest
import torch

from birdset_benchmark.models.vit_audio import ViTAudio


class TestViTAudio:
    """Test ViTAudio model with 1-channel patch embedding."""

    def test_model_instantiation(self):
        """Test that model can be instantiated."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )
        assert model is not None

    @pytest.mark.slow
    def test_model_pretrained_loading(self):
        """Test loading pretrained weights."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=True,
        )
        assert model is not None

    def test_model_forward_pass(self):
        """Test forward pass with single sample."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )
        model.eval()

        # Create dummy mel-spectrogram (batch=1, channels=1, n_mels=128, frames=500)
        x = torch.randn(1, 1, 128, 500)

        with torch.no_grad():
            logits = model(x)

        # Check output shape
        assert logits.shape == (1, 21), f"Expected (1, 21), got {logits.shape}"

        # Check no NaN or Inf
        assert not torch.isnan(logits).any(), "Output contains NaN"
        assert not torch.isinf(logits).any(), "Output contains Inf"

    def test_model_forward_batch(self):
        """Test forward pass with batch."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )
        model.eval()

        # Create batch of mel-spectrograms
        batch_size = 8
        x = torch.randn(batch_size, 1, 128, 500)

        with torch.no_grad():
            logits = model(x)

        # Check output shape
        assert logits.shape == (batch_size, 21), f"Expected ({batch_size}, 21), got {logits.shape}"

    def test_model_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [21, 100, 9736]:
            model = ViTAudio(
                num_classes=num_classes,
                img_size=(128, 500),
                patch_size=16,
                pretrained=False,
            )

            x = torch.randn(2, 1, 128, 500)

            with torch.no_grad():
                logits = model(x)

            assert logits.shape == (
                2,
                num_classes,
            ), f"Expected (2, {num_classes}), got {logits.shape}"

    def test_model_training_mode(self):
        """Test model in training mode (dropout enabled)."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
            drop_rate=0.1,
        )
        model.train()

        x = torch.randn(4, 1, 128, 500)

        # Forward pass should work in training mode
        logits = model(x)
        assert logits.shape == (4, 21)

        # Check gradients can be computed
        loss = logits.sum()
        loss.backward()

        # Check some gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_gpu(self):
        """Test model on GPU."""
        device = torch.device("cuda:0")

        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )
        model = model.to(device)
        model.eval()

        x = torch.randn(2, 1, 128, 500, device=device)

        with torch.no_grad():
            logits = model(x)

        assert logits.device == device, "Output not on GPU"
        assert logits.shape == (2, 21)

    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        num_params = sum(p.numel() for p in model.parameters())

        # ViT-Base should have ~86M parameters
        assert 80_000_000 < num_params < 100_000_000, f"Unexpected parameter count: {num_params}"

    def test_model_output_range(self):
        """Test that output logits are in reasonable range."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )
        model.eval()

        x = torch.randn(4, 1, 128, 500)

        with torch.no_grad():
            logits = model(x)

        # Logits should be roughly in [-10, 10] range for random init
        assert logits.min() > -20, f"Logits too negative: {logits.min()}"
        assert logits.max() < 20, f"Logits too positive: {logits.max()}"
