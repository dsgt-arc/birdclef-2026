"""PyTorch Lightning classifiers for BirdSet benchmark.

Contains LinearProbe models and custom metrics for multi-label classification.

Used by:
- Experiment 001: BirdSet Bioacoustics Benchmark (linear probing with BirdNET, Perch)

Adapted from dsgt-arc/birdclef-2025 for multi-label classification.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric


class MultiLabelTopKAccuracy(Metric):
    """Top-K accuracy for multi-label classification.

    For each sample, checks if at least one true label appears in the top-k predictions.
    This is the fraction of samples where the top-k predicted classes contain at least
    one true positive.

    Args:
        k: Number of top predictions to consider (e.g., 1 for top-1, 5 for top-5)
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            preds: Predicted probabilities [batch, num_classes]
            target: True multi-hot labels [batch, num_classes]
        """
        # Get top-k predicted class indices for each sample
        topk_indices = torch.topk(preds, k=self.k, dim=1).indices  # [batch, k]

        # For each sample, check if any of the top-k predictions match a true label
        for i in range(preds.shape[0]):
            # Get indices of true labels for this sample
            true_indices = torch.where(target[i] > 0)[0]

            # Check if any top-k prediction matches any true label
            if len(true_indices) > 0:  # Only consider samples with at least one label
                has_match = torch.any(torch.isin(topk_indices[i], true_indices))
                self.correct += has_match.int()
                self.total += 1

    def compute(self):
        """Compute the final metric."""
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total


class MultiLabelLinearProbe(pl.LightningModule):
    """Linear probe for multi-label bird species classification.

    Architecture:
        - Input: Pre-computed embeddings (e.g., 1024-dim from BirdNET)
        - Hidden: 512-dim with ReLU activation
        - Output: num_classes logits (BCEWithLogitsLoss includes sigmoid)

    Multi-label setup:
        - Loss: BCEWithLogitsLoss (binary cross-entropy per class)
        - Metrics: Multi-label F1, Average Precision
        - Labels: Multi-hot vectors [batch, num_classes]

    Args:
        input_dim: Embedding dimension (e.g., 1024 for BirdNET, 1280 for Perch)
        num_classes: Number of bird species classes
        hidden_dim: Hidden layer dimension (default: 512)
        lr: Learning rate (default: 1e-3)
        threshold: Classification threshold for binary predictions (default: 0.5)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        lr: float = 1e-3,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Multi-label loss (includes sigmoid internally)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.threshold = threshold
        self.num_classes = num_classes

        # Multi-label metrics
        # cmAP (Class-mean Average Precision) - BirdSet primary metric
        # This computes AP per class and averages them (macro average)
        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )

        # F1 scores
        self.train_f1_macro = torchmetrics.F1Score(
            task="multilabel", average="macro", num_labels=num_classes
        )
        self.train_f1_micro = torchmetrics.F1Score(
            task="multilabel", average="micro", num_labels=num_classes
        )
        self.val_f1_macro = torchmetrics.F1Score(
            task="multilabel", average="macro", num_labels=num_classes
        )
        self.val_f1_micro = torchmetrics.F1Score(
            task="multilabel", average="micro", num_labels=num_classes
        )

        # Top-K accuracy (for multi-label: any true label in top-k predictions)
        self.train_top1 = MultiLabelTopKAccuracy(k=1)
        self.train_top5 = MultiLabelTopKAccuracy(k=5)
        self.val_top1 = MultiLabelTopKAccuracy(k=1)
        self.val_top5 = MultiLabelTopKAccuracy(k=5)

    def forward(self, x):
        """Forward pass returns raw logits (before sigmoid)."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for multi-label classification."""
        x, y = batch  # x: [batch, input_dim], y: [batch, num_classes] multi-hot
        logits = self(x)

        # Loss computation
        loss = self.loss_fn(logits, y.float())

        # Convert logits to probabilities for metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        # Update metrics
        self.train_cmap.update(probs, y.int())
        self.train_f1_macro.update(preds, y.int())
        self.train_f1_micro.update(preds, y.int())
        self.train_top1.update(probs, y.int())
        self.train_top5.update(probs, y.int())

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        self.log("train_cmap", self.train_cmap.compute(), prog_bar=True)
        self.log("train_f1_macro", self.train_f1_macro.compute(), prog_bar=True)
        self.log("train_f1_micro", self.train_f1_micro.compute(), prog_bar=True)
        self.log("train_top1_acc", self.train_top1.compute(), prog_bar=False)
        self.log("train_top5_acc", self.train_top5.compute(), prog_bar=False)

        self.train_cmap.reset()
        self.train_f1_macro.reset()
        self.train_f1_micro.reset()
        self.train_top1.reset()
        self.train_top5.reset()

    def validation_step(self, batch, batch_idx):
        """Validation step for multi-label classification."""
        x, y = batch
        logits = self(x)

        # Loss computation
        loss = self.loss_fn(logits, y.float())

        # Convert logits to probabilities for metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        # Update metrics
        self.val_cmap.update(probs, y.int())
        self.val_f1_macro.update(preds, y.int())
        self.val_f1_micro.update(preds, y.int())
        self.val_top1.update(probs, y.int())
        self.val_top5.update(probs, y.int())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        self.log("val_cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro.compute(), prog_bar=True)
        self.log("val_f1_micro", self.val_f1_micro.compute(), prog_bar=True)
        self.log("val_top1_acc", self.val_top1.compute(), prog_bar=True)
        self.log("val_top5_acc", self.val_top5.compute(), prog_bar=True)

        self.val_cmap.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()
        self.val_top1.reset()
        self.val_top5.reset()

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx):
        """Prediction step returning probabilities and binary predictions."""
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        return {"probabilities": probs, "predictions": preds, "logits": logits}


class LinearProbe(pl.LightningModule):
    """Simple linear probe (no hidden layer) for comparison.

    This is a true linear probe - just a single linear layer on top of embeddings.
    Useful for evaluating embedding quality without confounding factors.

    Args:
        input_dim: Embedding dimension
        num_classes: Number of bird species classes
        lr: Learning rate (default: 1e-3)
        threshold: Classification threshold (default: 0.5)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Single linear layer
        self.model = nn.Linear(input_dim, num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.threshold = threshold
        self.num_classes = num_classes

        # Metrics (same as MultiLabelLinearProbe)
        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.train_f1_macro = torchmetrics.F1Score(
            task="multilabel", average="macro", num_labels=num_classes
        )
        self.train_f1_micro = torchmetrics.F1Score(
            task="multilabel", average="micro", num_labels=num_classes
        )
        self.val_f1_macro = torchmetrics.F1Score(
            task="multilabel", average="macro", num_labels=num_classes
        )
        self.val_f1_micro = torchmetrics.F1Score(
            task="multilabel", average="micro", num_labels=num_classes
        )
        self.train_top1 = MultiLabelTopKAccuracy(k=1)
        self.train_top5 = MultiLabelTopKAccuracy(k=5)
        self.val_top1 = MultiLabelTopKAccuracy(k=1)
        self.val_top5 = MultiLabelTopKAccuracy(k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.train_cmap.update(probs, y.int())
        self.train_f1_macro.update(preds, y.int())
        self.train_f1_micro.update(preds, y.int())
        self.train_top1.update(probs, y.int())
        self.train_top5.update(probs, y.int())

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_cmap", self.train_cmap.compute(), prog_bar=True)
        self.log("train_f1_macro", self.train_f1_macro.compute(), prog_bar=True)
        self.log("train_f1_micro", self.train_f1_micro.compute(), prog_bar=True)
        self.log("train_top1_acc", self.train_top1.compute(), prog_bar=False)
        self.log("train_top5_acc", self.train_top5.compute(), prog_bar=False)

        self.train_cmap.reset()
        self.train_f1_macro.reset()
        self.train_f1_micro.reset()
        self.train_top1.reset()
        self.train_top5.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.val_cmap.update(probs, y.int())
        self.val_f1_macro.update(preds, y.int())
        self.val_f1_micro.update(preds, y.int())
        self.val_top1.update(probs, y.int())
        self.val_top5.update(probs, y.int())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val_cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro.compute(), prog_bar=True)
        self.log("val_f1_micro", self.val_f1_micro.compute(), prog_bar=True)
        self.log("val_top1_acc", self.val_top1.compute(), prog_bar=True)
        self.log("val_top5_acc", self.val_top5.compute(), prog_bar=True)

        self.val_cmap.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()
        self.val_top1.reset()
        self.val_top5.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        return {"probabilities": probs, "predictions": preds, "logits": logits}
