"""PyTorch Lightning module for fine-tuning pretrained audio encoders.

Supports frozen linear probing and staged unfreezing for EAT/BEATs encoders.
Data flow: waveforms -> encoder -> classification head -> logits.

Separate from SupervisedViTModule because encoders consume raw waveforms
(not mel-spectrograms) and handle their own feature extraction internally.
"""

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from birdclef.classifiers import MultiLabelTopKAccuracy

logger = logging.getLogger(__name__)


class PretrainedFinetuneModule(pl.LightningModule):
    """Fine-tune a pretrained audio encoder with a classification head.

    Supports two training modes:
      1. Linear probe: encoder frozen, only head trains (freeze_encoder=True)
      2. Full fine-tune: encoder unfreezes at unfreeze_epoch, with lower LR

    Args:
        encoder: Pretrained audio encoder (EATEncoder, BEATsEncoder, etc.)
        num_classes: Number of output classes
        embed_dim: Encoder embedding dimension (default: 768 for EAT/BEATs)
        hidden_dim: Hidden layer dimension (0 = no hidden layer, linear probe)
        dropout: Dropout rate for hidden layer
        lr: Learning rate for classification head
        encoder_lr: Learning rate for encoder (typically 10x lower than head)
        weight_decay: AdamW weight decay
        max_epochs: Maximum number of epochs for cosine annealing
        warmup_epochs: Number of warmup epochs
        freeze_encoder: Whether to freeze encoder initially
        unfreeze_epoch: Epoch at which to unfreeze encoder (0 = never unfreeze)
        threshold: Classification threshold for binary predictions
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        embed_dim: int = 768,
        hidden_dim: int = 0,
        dropout: float = 0.1,
        lr: float = 1e-3,
        encoder_lr: float = 1e-5,
        weight_decay: float = 0.01,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        freeze_encoder: bool = True,
        unfreeze_epoch: int = 0,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.num_classes = num_classes
        self.threshold = threshold

        # Classification head
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        # Multi-label loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Freeze encoder if requested
        if freeze_encoder and hasattr(encoder, "freeze"):
            encoder.freeze()
            logger.info("Encoder frozen for linear probing")

        # --- Metrics ---
        # Training
        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.train_auroc = torchmetrics.AUROC(
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

        # Validation
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_auroc = torchmetrics.AUROC(
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass: waveform -> encoder -> head -> logits.

        Args:
            waveform: (batch, samples) at 16kHz

        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.encoder(waveform)  # (batch, embed_dim)
        logits = self.head(features)  # (batch, num_classes)
        return logits

    def on_train_epoch_start(self):
        """Unfreeze encoder at the configured epoch."""
        unfreeze_epoch = self.hparams.unfreeze_epoch
        if (
            unfreeze_epoch > 0
            and self.current_epoch == unfreeze_epoch
            and hasattr(self.encoder, "unfreeze")
        ):
            self.encoder.unfreeze()
            logger.info(f"Encoder unfrozen at epoch {self.current_epoch}")

    def training_step(self, batch, batch_idx):
        waveform, labels = batch
        logits = self(waveform)

        loss = self.criterion(logits, labels.float())

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.train_cmap.update(probs, labels.int())
        self.train_auroc.update(probs, labels.int())
        self.train_f1_macro.update(preds, labels.int())
        self.train_f1_micro.update(preds, labels.int())
        self.train_top1.update(probs, labels.int())
        self.train_top5.update(probs, labels.int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=False)
        self.log("train/f1_macro", self.train_f1_macro.compute(), prog_bar=True)
        self.log("train/f1_micro", self.train_f1_micro.compute(), prog_bar=False)
        self.log("train/top1_acc", self.train_top1.compute(), prog_bar=False)
        self.log("train/top5_acc", self.train_top5.compute(), prog_bar=False)

        self.train_cmap.reset()
        self.train_auroc.reset()
        self.train_f1_macro.reset()
        self.train_f1_micro.reset()
        self.train_top1.reset()
        self.train_top5.reset()

    def validation_step(self, batch, batch_idx):
        waveform, labels = batch
        logits = self(waveform)

        loss = self.criterion(logits, labels.float())

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.val_cmap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())
        self.val_f1_macro.update(preds, labels.int())
        self.val_f1_micro.update(preds, labels.int())
        self.val_top1.update(probs, labels.int())
        self.val_top5.update(probs, labels.int())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro.compute(), prog_bar=True)
        self.log("val/f1_micro", self.val_f1_micro.compute(), prog_bar=False)
        self.log("val/top1_acc", self.val_top1.compute(), prog_bar=True)
        self.log("val/top5_acc", self.val_top5.compute(), prog_bar=True)

        self.val_cmap.reset()
        self.val_auroc.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()
        self.val_top1.reset()
        self.val_top5.reset()

    def configure_optimizers(self):
        """Configure AdamW with separate LR for encoder and head.

        When the encoder is unfrozen, uses two param groups:
          - Head parameters at self.hparams.lr
          - Encoder parameters at self.hparams.encoder_lr (typically 10x lower)
        """
        # Head parameters always train
        param_groups = [
            {"params": self.head.parameters(), "lr": self.hparams.lr},
        ]

        # Add encoder params if any are trainable
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.hparams.encoder_lr},
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-7,
        )

        # Linear warmup for first N epochs
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs,
            )
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
        waveform = batch if isinstance(batch, torch.Tensor) else batch[0]
        logits = self(waveform)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        return {"probabilities": probs, "predictions": preds, "logits": logits}
