"""PyTorch Lightning module for SSAMBA multi-label classification on BirdSet.

Supports freeze/unfreeze of backbone for pre-trained transfer learning.
"""

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from .spec_augment import SpecAugment

logger = logging.getLogger(__name__)


class SSAMBALightningModule(pl.LightningModule):
    """Lightning module for SSAMBA training with label smoothing and SpecAugment.

    Args:
        model: SSAMBAClassifier instance
        num_classes: Number of output classes
        lr: Learning rate
        weight_decay: AdamW weight decay
        max_epochs: Max epochs for cosine annealing
        warmup_epochs: Linear warmup epochs
        label_smoothing: Label smoothing factor (0 = none)
        threshold: Classification threshold for binary predictions
        augmentation: Dict of SpecAugment params (or None to disable)
        freeze_epochs: Number of epochs to freeze backbone (0 = no freezing)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 5e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        label_smoothing: float = 0.1,
        threshold: float = 0.5,
        augmentation: dict | None = None,
        freeze_epochs: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.threshold = threshold
        self.label_smoothing = label_smoothing
        self.freeze_epochs = freeze_epochs
        self._backbone_frozen = False

        self.criterion = nn.BCEWithLogitsLoss()

        # SpecAugment (training only)
        if augmentation:
            self.spec_augment = SpecAugment(**augmentation)
        else:
            self.spec_augment = None

        # Metrics — train
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

        # Metrics — val
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

    def _freeze_backbone(self):
        """Freeze backbone Mamba layers and norms, leave patch_embed and head trainable."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Keep trainable: patch_embed, head, pos_embed, cls_token
            if any(k in name for k in ["head.", "patch_embed.", "pos_embed", "cls_token"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Backbone frozen: {frozen_count} params frozen, "
            f"{trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)"
        )
        self._backbone_frozen = True

    def _unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Backbone unfrozen: all {trainable:,} params trainable")
        self._backbone_frozen = False

    def on_train_epoch_start(self):
        if self.freeze_epochs > 0:
            if self.current_epoch < self.freeze_epochs and not self._backbone_frozen:
                self._freeze_backbone()
            elif self.current_epoch >= self.freeze_epochs and self._backbone_frozen:
                self._unfreeze_backbone()

    def forward(self, x, padding_mask=None):
        return self.model(x, padding_mask=padding_mask)

    def _smooth_labels(self, targets):
        """Apply label smoothing: targets * (1 - eps) + eps/2."""
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            return targets * (1 - eps) + eps / 2
        return targets

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        mel_specs, labels, padding_mask = batch

        if self.spec_augment is not None:
            mel_specs = self.spec_augment(mel_specs)

        logits = self(mel_specs, padding_mask=padding_mask)

        smoothed = self._smooth_labels(labels)
        loss = self.criterion(logits, smoothed)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.train_cmap.update(probs, labels.int())
        self.train_auroc.update(probs, labels.int())
        self.train_f1_macro.update(preds, labels.int())
        self.train_f1_micro.update(preds, labels.int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.log("train/auroc", self.train_auroc.compute())
        self.log("train/f1_macro", self.train_f1_macro.compute(), prog_bar=True)
        self.log("train/f1_micro", self.train_f1_micro.compute())

        self.train_cmap.reset()
        self.train_auroc.reset()
        self.train_f1_macro.reset()
        self.train_f1_micro.reset()

    def validation_step(self, batch, batch_idx):
        mel_specs, labels, padding_mask = batch
        logits = self(mel_specs, padding_mask=padding_mask)

        loss = self.criterion(logits, labels.float())

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.val_cmap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())
        self.val_f1_macro.update(preds, labels.int())
        self.val_f1_micro.update(preds, labels.int())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro.compute(), prog_bar=True)
        self.log("val/f1_micro", self.val_f1_micro.compute())

        self.val_cmap.reset()
        self.val_auroc.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()

    def configure_optimizers(self):
        # Only optimize trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        if self.hparams.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.hparams.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[self.hparams.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
