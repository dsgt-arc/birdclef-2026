"""Lightning training module for exp 102. Vendored from exp 009 with modifications.

Changes from exp 009:
- Uses local FocalBCELoss and SpecAugment (no exp 009 import)
- Supports sample_weight from activity scores (optional)
"""

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from .losses import FocalBCELoss
from .spec_augment import SpecAugment

logger = logging.getLogger(__name__)


class TrainingModule(pl.LightningModule):
    """Lightning module for clip-level fine-tuning with focal BCE.

    Args:
        model: EfficientNetClassifier instance
        num_classes: Number of output classes
        lr: Learning rate
        weight_decay: AdamW weight decay
        max_epochs: Max epochs for cosine annealing
        warmup_epochs: Linear warmup epochs
        augmentation: Dict of SpecAugment params
        gamma: Focal gamma for clip loss
        use_activity_weights: If True, use activity_scores from batch as sample weights
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 50,
        warmup_epochs: int = 3,
        augmentation: dict | None = None,
        gamma: float = 2.0,
        use_activity_weights: bool = False,
        from_logits: bool = True,
        class_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weight"])

        self.model = model
        self.num_classes = num_classes
        self.use_activity_weights = use_activity_weights
        self.from_logits = from_logits

        # avg pooling → from_logits=True (model returns logits)
        # attention pooling → from_logits=False (model returns probabilities)
        self.clip_loss = FocalBCELoss(gamma=gamma, from_logits=from_logits,
                                      class_weight=class_weight)

        if augmentation:
            self.spec_augment = SpecAugment(**augmentation)
        else:
            self.spec_augment = None

        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_auroc = torchmetrics.AUROC(
            task="multilabel", num_labels=num_classes, average="macro"
        )

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float("inf")
        )
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

    def _to_probs(self, output):
        """Convert model output to probabilities for metrics."""
        if self.from_logits:
            return torch.sigmoid(output)
        else:
            return output  # already probabilities (attention pooling)

    def training_step(self, batch, batch_idx):
        clip_labels = batch["clip_labels"]

        output = self.model(
            waveforms=batch.get("waveform"),
            mel=batch.get("mel"),
            spec_augment=self.spec_augment if self.training else None,
        )

        # Optional activity-based sample weighting
        sample_weight = None
        if self.use_activity_weights and "activity_scores" in batch:
            sample_weight = batch["activity_scores"]

        loss = self.clip_loss(output, clip_labels, sample_weight=sample_weight)

        self.train_cmap.update(self._to_probs(output), clip_labels.round().int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.train_cmap.reset()

    def validation_step(self, batch, batch_idx):
        clip_labels = batch["clip_labels"]

        output = self.model(
            waveforms=batch.get("waveform"),
            mel=batch.get("mel"),
        )
        loss = self.clip_loss(output, clip_labels)

        probs = self._to_probs(output)
        self.val_cmap.update(probs, clip_labels.int())
        self.val_auroc.update(probs, clip_labels.int())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.val_cmap.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        cosine_epochs = self.hparams.max_epochs - self.hparams.warmup_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6
        )

        if self.hparams.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
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
