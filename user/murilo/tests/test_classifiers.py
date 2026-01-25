"""Test PyTorch Lightning classifiers for linear probing."""

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from birdset_benchmark.models import (
    MultiLabelLinearProbe,
    LinearProbe,
    MultiLabelTopKAccuracy,
)


class TestMultiLabelMetrics:
    """Test custom multi-label metrics."""

    def test_topk_accuracy_top1(self):
        """Test Top-1 accuracy for multi-label."""
        metric = MultiLabelTopKAccuracy(k=1)

        # Example: 3 samples, 5 classes
        # Sample 1: true=[1,0,1,0,0], top pred=class 0 (correct!)
        # Sample 2: true=[0,1,0,0,0], top pred=class 2 (wrong)
        # Sample 3: true=[0,0,0,1,1], top pred=class 3 (correct!)
        preds = torch.tensor([
            [0.9, 0.2, 0.3, 0.1, 0.05],  # Top-1: class 0
            [0.1, 0.3, 0.8, 0.2, 0.1],   # Top-1: class 2
            [0.1, 0.2, 0.3, 0.7, 0.5],   # Top-1: class 3
        ])
        target = torch.tensor([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
        ])

        metric.update(preds, target)
        accuracy = metric.compute()

        # 2 out of 3 correct (samples 1 and 3)
        assert accuracy.item() == pytest.approx(2/3, abs=0.01)

    def test_topk_accuracy_top5(self):
        """Test Top-5 accuracy for multi-label."""
        metric = MultiLabelTopKAccuracy(k=5)

        # With k=5 and only 5 classes, all predictions contain true labels
        preds = torch.rand(10, 5)
        target = torch.randint(0, 2, (10, 5))

        metric.update(preds, target)
        accuracy = metric.compute()

        # Should be 1.0 since top-5 includes all classes
        assert accuracy.item() == 1.0


class TestMultiLabelLinearProbe:
    """Test multi-label linear probe classifier."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = MultiLabelLinearProbe(
            input_dim=1024,
            num_classes=50,
            hidden_dim=512,
            lr=1e-3
        )
        assert model is not None
        assert model.num_classes == 50

    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        model = MultiLabelLinearProbe(input_dim=1024, num_classes=50)
        x = torch.randn(16, 1024)
        logits = model(x)

        assert logits.shape == (16, 50)

    def test_training_step(self):
        """Test training step executes without errors."""
        model = MultiLabelLinearProbe(input_dim=128, num_classes=10)

        # Create dummy batch
        x = torch.randn(16, 128)
        y = torch.randint(0, 2, (16, 10))  # Multi-hot labels
        batch = (x, y)

        # Run training step
        loss = model.training_step(batch, batch_idx=0)

        assert loss is not None
        assert loss.item() > 0

    @pytest.mark.slow
    def test_full_training_loop(self):
        """Test full training loop with dummy data."""
        # Create dummy dataset
        X = torch.randn(100, 128)
        y = torch.randint(0, 2, (100, 10)).float()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Initialize model
        model = MultiLabelLinearProbe(input_dim=128, num_classes=10, lr=1e-2)

        # Train for 2 epochs
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Check that metrics were logged
        assert trainer.logged_metrics is not None


class TestLinearProbe:
    """Test pure linear probe (no hidden layer)."""

    def test_model_initialization(self):
        """Test linear probe can be initialized."""
        model = LinearProbe(input_dim=1024, num_classes=50)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        model = LinearProbe(input_dim=1024, num_classes=50)
        x = torch.randn(16, 1024)
        logits = model(x)

        assert logits.shape == (16, 50)

    def test_model_is_linear(self):
        """Verify model is truly linear (no hidden layers)."""
        model = LinearProbe(input_dim=1024, num_classes=50)

        # Should only have one linear layer
        assert isinstance(model.model, torch.nn.Linear)
        assert model.model.in_features == 1024
        assert model.model.out_features == 50

    def test_predict_step(self):
        """Test prediction step returns probabilities and predictions."""
        model = LinearProbe(input_dim=128, num_classes=10)
        x = torch.randn(16, 128)

        result = model.predict_step(x, batch_idx=0)

        assert "probabilities" in result
        assert "predictions" in result
        assert "logits" in result
        assert result["probabilities"].shape == (16, 10)
        assert result["predictions"].shape == (16, 10)
        assert result["logits"].shape == (16, 10)
