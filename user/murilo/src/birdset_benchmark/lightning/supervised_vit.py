"""PyTorch Lightning module for supervised ViT training on BirdSet.

Used by: Experiment 002 (Supervised ViT Baseline)

Following patterns from classifiers.py (MultiLabelLinearProbe) for metrics and training.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from birdset_benchmark.classifiers import MultiLabelTopKAccuracy


class SupervisedViTModule(pl.LightningModule):
    """PyTorch Lightning module for supervised ViT-Base training.

    Args:
        model: ViTAudio model instance
        num_classes: Number of output classes
        lr: Learning rate (default: 1e-4)
        weight_decay: AdamW weight decay (default: 0.01)
        max_epochs: Maximum number of epochs for cosine annealing (default: 50)
        warmup_epochs: Number of warmup epochs (default: 5)
        threshold: Classification threshold for binary predictions (default: 0.5)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.threshold = threshold

        # Multi-label loss (includes sigmoid internally)
        self.criterion = nn.BCEWithLogitsLoss()

        # Training metrics
        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.train_f1_macro = torchmetrics.F1Score(
            task="multilabel", average="macro", num_labels=num_classes
        )
        self.train_f1_micro = torchmetrics.F1Score(
            task="multilabel", average="micro", num_labels=num_classes
        )
        self.train_top1 = MultiLabelTopKAccuracy(k=1)
        self.train_top5 = MultiLabelTopKAccuracy(k=5)

        # Validation metrics
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_f1_macro = torchmetrics.F1Score(
            task="multilabel", average="macro", num_labels=num_classes
        )
        self.val_f1_micro = torchmetrics.F1Score(
            task="multilabel", average="micro", num_labels=num_classes
        )
        self.val_top1 = MultiLabelTopKAccuracy(k=1)
        self.val_top5 = MultiLabelTopKAccuracy(k=5)

    def forward(self, x):
        """Forward pass returns raw logits (before sigmoid)."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for multi-label classification.

        Args:
            batch: Tuple of (mel_spectrograms, labels)
                - mel_spectrograms: (batch, 1, n_mels, target_frames)
                - labels: (batch, num_classes) multi-hot
        """
        mel_specs, labels = batch
        logits = self(mel_specs)

        # Compute loss
        loss = self.criterion(logits, labels.float())

        # Convert logits to probabilities for metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        # Update metrics
        self.train_cmap.update(probs, labels.int())
        self.train_f1_macro.update(preds, labels.int())
        self.train_f1_micro.update(preds, labels.int())
        self.train_top1.update(probs, labels.int())
        self.train_top5.update(probs, labels.int())

        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.log("train/f1_macro", self.train_f1_macro.compute(), prog_bar=True)
        self.log("train/f1_micro", self.train_f1_micro.compute(), prog_bar=False)
        self.log("train/top1_acc", self.train_top1.compute(), prog_bar=False)
        self.log("train/top5_acc", self.train_top5.compute(), prog_bar=False)

        # Reset metrics
        self.train_cmap.reset()
        self.train_f1_macro.reset()
        self.train_f1_micro.reset()
        self.train_top1.reset()
        self.train_top5.reset()

    def validation_step(self, batch, batch_idx):
        """Validation step for multi-label classification."""
        mel_specs, labels = batch
        logits = self(mel_specs)

        # Compute loss
        loss = self.criterion(logits, labels.float())

        # Convert logits to probabilities for metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        # Update metrics
        self.val_cmap.update(probs, labels.int())
        self.val_f1_macro.update(preds, labels.int())
        self.val_f1_micro.update(preds, labels.int())
        self.val_top1.update(probs, labels.int())
        self.val_top5.update(probs, labels.int())

        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro.compute(), prog_bar=True)
        self.log("val/f1_micro", self.val_f1_micro.compute(), prog_bar=False)
        self.log("val/top1_acc", self.val_top1.compute(), prog_bar=True)
        self.log("val/top5_acc", self.val_top5.compute(), prog_bar=True)

        # Reset metrics
        self.val_cmap.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()
        self.val_top1.reset()
        self.val_top5.reset()

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing and warmup."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )

        # Linear warmup for first N epochs
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of lr
                end_factor=1.0,  # Ramp up to full lr
                total_iters=self.hparams.warmup_epochs,
            )

            # Sequential scheduler: warmup then cosine annealing
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        """Prediction step returning probabilities and binary predictions."""
        mel_specs = batch if isinstance(batch, torch.Tensor) else batch[0]
        logits = self(mel_specs)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        return {"probabilities": probs, "predictions": preds, "logits": logits}
