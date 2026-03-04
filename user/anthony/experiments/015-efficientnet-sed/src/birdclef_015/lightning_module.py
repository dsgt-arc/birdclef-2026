"""PyTorch Lightning module for EfficientNet SED with dual loss."""

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from birdclef_009.utils.losses import FocalBCELoss
from birdclef_009.utils.spec_augment import SpecAugment

logger = logging.getLogger(__name__)


class SEDLightningModule(pl.LightningModule):
    """Lightning module for SED training with dual clip + frame loss.

    Loss = clip_loss + alpha * frame_loss
    - frame_loss: FocalBCE(frame_logits, frame_labels, mask=frame_mask) — logits mode
    - clip_loss: FocalBCE(clip_pred, clip_labels) — probability mode

    Args:
        model: EfficientNetSED instance
        num_classes: Number of output classes
        lr: Learning rate
        weight_decay: AdamW weight decay
        max_epochs: Max epochs for cosine annealing
        warmup_epochs: Linear warmup epochs
        alpha: Weight for frame loss (default 1.0)
        gamma_clip: Focal gamma for clip loss
        gamma_frame: Focal gamma for frame loss
        augmentation: Dict of SpecAugment params
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 30,
        warmup_epochs: int = 0,
        alpha: float = 1.0,
        gamma_clip: float = 2.0,
        gamma_frame: float = 2.0,
        augmentation: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.alpha = alpha

        # Clip loss operates on probabilities (post-sigmoid from attention pooling)
        self.clip_loss = FocalBCELoss(gamma=gamma_clip, from_logits=False)
        # Frame loss operates on raw logits
        self.frame_loss = FocalBCELoss(gamma=gamma_frame, from_logits=True)

        if augmentation:
            self.spec_augment = SpecAugment(**augmentation)
        else:
            self.spec_augment = None

        # Metrics — train
        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        # Metrics — val
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_auroc = torchmetrics.AUROC(
            task="multilabel", num_labels=num_classes, average="macro"
        )

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        waveforms = batch["waveform"]
        clip_labels = batch["clip_labels"]
        frame_labels = batch["frame_labels"]  # (B, 5, C)
        frame_mask = batch["frame_mask"]      # (B, 5)

        clip_pred, frame_logits = self.model(
            waveforms,
            spec_augment=self.spec_augment if self.training else None,
        )

        # Dual loss
        loss_clip = self.clip_loss(clip_pred, clip_labels)
        loss_frame = self.frame_loss(frame_logits, frame_labels, mask=frame_mask)
        loss = loss_clip + self.alpha * loss_frame

        self.train_cmap.update(clip_pred, clip_labels.round().int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_clip", loss_clip, on_step=True, on_epoch=True)
        self.log("train/loss_frame", loss_frame, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.train_cmap.reset()

    def validation_step(self, batch, batch_idx):
        waveforms = batch["waveform"]
        clip_labels = batch["clip_labels"]

        # Val uses hard labels — only clip-level evaluation
        clip_pred, _ = self.model(waveforms)

        loss = self.clip_loss(clip_pred, clip_labels)

        self.val_cmap.update(clip_pred, clip_labels.int())
        self.val_auroc.update(clip_pred, clip_labels.int())

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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        if self.hparams.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=self.hparams.warmup_epochs,
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
