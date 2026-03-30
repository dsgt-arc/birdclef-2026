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
        clipwise_weight: float = 0.5,
        framewise_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weight"])

        self.model = model
        self.num_classes = num_classes
        self.use_activity_weights = use_activity_weights
        self.from_logits = from_logits
        self.clipwise_weight = clipwise_weight
        self.framewise_weight = framewise_weight

        # avg pooling → from_logits=True (model returns logits)
        # attention pooling → from_logits=False (model returns probabilities)
        # sed pooling → from_logits=True (model returns logits in dict)
        self.clip_loss = FocalBCELoss(gamma=gamma, from_logits=from_logits,
                                      class_weight=class_weight)
        # Framewise max loss for SED dual supervision (same loss, always from logits)
        self.frame_loss = FocalBCELoss(gamma=gamma, from_logits=True,
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
        # Collect val predictions for Kaggle-style AUROC (excludes absent species)
        self._val_probs = []
        self._val_targets = []
        # Auxiliary soundscape validation (dataloader_idx=1)
        self._sc_probs = []
        self._sc_targets = []

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

    def _unpack_output(self, output):
        """Unpack model output — handles both tensor and dict (SED mode)."""
        if isinstance(output, dict):
            return output["clipwise_logits"], output["framewise_logits"]
        return output, None

    def training_step(self, batch, batch_idx):
        clip_labels = batch["clip_labels"]

        output = self.model(
            waveforms=batch.get("waveform"),
            mel=batch.get("mel"),
            spec_augment=self.spec_augment if self.training else None,
        )

        clip_output, frame_logits = self._unpack_output(output)

        # Optional activity-based sample weighting
        sample_weight = None
        if self.use_activity_weights and "activity_scores" in batch:
            sample_weight = batch["activity_scores"]

        clip_loss = self.clip_loss(clip_output, clip_labels, sample_weight=sample_weight)

        if frame_logits is not None:
            # Dual supervision: framewise max should also predict clip labels
            frame_max = frame_logits.max(dim=-1).values  # (B, C)
            f_loss = self.frame_loss(frame_max, clip_labels, sample_weight=sample_weight)
            loss = self.clipwise_weight * clip_loss + self.framewise_weight * f_loss
            self.log("train/clip_loss", clip_loss, on_step=False, on_epoch=True)
            self.log("train/frame_loss", f_loss, on_step=False, on_epoch=True)
        else:
            loss = clip_loss

        self.train_cmap.update(self._to_probs(clip_output), clip_labels.round().int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.train_cmap.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        clip_labels = batch["clip_labels"]

        output = self.model(
            waveforms=batch.get("waveform"),
            mel=batch.get("mel"),
        )

        clip_output, frame_logits = self._unpack_output(output)
        probs = self._to_probs(clip_output)

        if dataloader_idx == 0:
            # Primary focal validation
            clip_loss = self.clip_loss(clip_output, clip_labels)
            if frame_logits is not None:
                frame_max = frame_logits.max(dim=-1).values
                f_loss = self.frame_loss(frame_max, clip_labels)
                loss = self.clipwise_weight * clip_loss + self.framewise_weight * f_loss
            else:
                loss = clip_loss

            self.val_cmap.update(probs, clip_labels.int())
            self._val_probs.append(probs.detach().cpu())
            self._val_targets.append(clip_labels.int().detach().cpu())
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            # Auxiliary soundscape validation (monitoring only)
            self._sc_probs.append(probs.detach().cpu())
            self._sc_targets.append(clip_labels.int().detach().cpu())

    def _compute_auroc(self, all_probs, all_targets):
        """Kaggle-style macro AUROC, excluding species with no positive labels."""
        col_sums = all_targets.sum(axis=0)
        scored_cols = col_sums > 0
        if scored_cols.sum() > 0:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(
                all_targets[:, scored_cols],
                all_probs[:, scored_cols],
                average="macro",
            )
        return 0.0

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.val_cmap.reset()

        # Focal AUROC
        if self._val_probs:
            all_probs = torch.cat(self._val_probs, dim=0).numpy()
            all_targets = torch.cat(self._val_targets, dim=0).numpy()
            self._val_probs.clear()
            self._val_targets.clear()
            self.log("val/auroc", self._compute_auroc(all_probs, all_targets), prog_bar=True)

        # Soundscape AUROC (auxiliary, monitoring only)
        if self._sc_probs:
            sc_probs = torch.cat(self._sc_probs, dim=0).numpy()
            sc_targets = torch.cat(self._sc_targets, dim=0).numpy()
            self._sc_probs.clear()
            self._sc_targets.clear()
            self.log("val/sc_auroc", self._compute_auroc(sc_probs, sc_targets), prog_bar=True)

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
