"""Test SupervisedViTModule for Experiment 002."""

import pytest
import torch

from birdset_benchmark.lightning.supervised_vit import SupervisedViTModule
from birdset_benchmark.models.vit_audio import ViTAudio


class TestSupervisedViTModule:
    """Test SupervisedViTModule with multi-label metrics."""

    def test_module_instantiation(self):
        """Test that Lightning module can be instantiated."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
            max_epochs=50,
            warmup_epochs=5,
        )

        assert pl_module is not None
        assert pl_module.num_classes == 21
        assert pl_module.lr == 1e-4

    def test_module_forward(self):
        """Test forward pass through Lightning module."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )

        # Create dummy input
        x = torch.randn(2, 1, 128, 500)

        with torch.no_grad():
            logits = pl_module(x)

        assert logits.shape == (2, 21)

    def test_module_training_step(self):
        """Test training step."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )

        # Create dummy batch
        mel_specs = torch.randn(4, 1, 128, 500)
        labels = torch.zeros(4, 21)
        labels[:, 0] = 1  # Set first class as positive
        batch = (mel_specs, labels)

        # Training step
        loss = pl_module.training_step(batch, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_module_validation_step(self):
        """Test validation step."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )

        # Create dummy batch
        mel_specs = torch.randn(4, 1, 128, 500)
        labels = torch.zeros(4, 21)
        labels[:, 0] = 1
        batch = (mel_specs, labels)

        # Validation step
        loss = pl_module.validation_step(batch, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_module_configure_optimizers(self):
        """Test optimizer configuration."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
            max_epochs=50,
            warmup_epochs=5,
        )

        # Get optimizer config
        opt_config = pl_module.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

        optimizer = opt_config["optimizer"]
        scheduler_config = opt_config["lr_scheduler"]

        # Check optimizer type
        assert isinstance(optimizer, torch.optim.AdamW)

        # Check scheduler exists
        assert "scheduler" in scheduler_config

    def test_module_loss_decreases(self):
        """Test that loss decreases with training on dummy data."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-3,  # Higher LR for faster convergence
            weight_decay=0.0,
        )
        pl_module.train()

        # Create fixed dummy batch
        torch.manual_seed(42)
        mel_specs = torch.randn(8, 1, 128, 500)
        labels = torch.zeros(8, 21)
        labels[:, 0] = 1
        batch = (mel_specs, labels)

        # Get optimizer
        opt_config = pl_module.configure_optimizers()
        optimizer = opt_config["optimizer"]

        # Train for a few steps
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            loss = pl_module.training_step(batch, batch_idx=step)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_module_gpu(self):
        """Test module on GPU."""
        device = torch.device("cuda:0")

        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )
        pl_module = pl_module.to(device)

        # Create batch on GPU
        mel_specs = torch.randn(4, 1, 128, 500, device=device)
        labels = torch.zeros(4, 21, device=device)
        labels[:, 0] = 1
        batch = (mel_specs, labels)

        # Should work on GPU
        loss = pl_module.training_step(batch, batch_idx=0)
        assert loss.device == device

    def test_module_metrics_computed(self):
        """Test that metrics are computed during validation."""
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )
        pl_module.eval()

        # Create dummy batch
        mel_specs = torch.randn(4, 1, 128, 500)
        labels = torch.zeros(4, 21)
        labels[:, 0] = 1
        batch = (mel_specs, labels)

        # Validation step
        with torch.no_grad():
            pl_module.validation_step(batch, batch_idx=0)

        # Metrics should be logged (stored in module)
        # This is a basic check - full metric validation requires trainer
        assert True  # Passes if no errors

    def test_module_different_num_classes(self):
        """Test module with different number of classes."""
        for num_classes in [21, 100, 9736]:
            model = ViTAudio(
                num_classes=num_classes,
                img_size=(128, 500),
                patch_size=16,
                pretrained=False,
            )

            pl_module = SupervisedViTModule(
                model=model,
                num_classes=num_classes,
                lr=1e-4,
                weight_decay=0.01,
            )

            # Create batch
            mel_specs = torch.randn(2, 1, 128, 500)
            labels = torch.zeros(2, num_classes)
            labels[:, 0] = 1
            batch = (mel_specs, labels)

            # Should work for any num_classes
            loss = pl_module.training_step(batch, batch_idx=0)
            assert loss.item() > 0
