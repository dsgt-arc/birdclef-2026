"""SED fine-tuning module with dual cross-entropy loss.

Architecture: encoder.forward_features() → AttHead → dual logits (clipwise + framewise).

Training loss: CE on clipwise + CE on framewise-max, with optional mixup augmentation.
Supports both spectrogram-domain mixup (v1/v2) and raw waveform mixup (v3b).

Inference: 0.75 * sigmoid(clipwise) + 0.25 * sigmoid(framewise_max).

Data flow:
  - Default: mel spectrogram (CPU) → backbone features → SED head → dual logits.
  - With mel_transform + waveform_mixup: raw waveform → mixup → mel (GPU) → backbone → SED head.
  - With mel_transform (no waveform_mixup): raw waveform → mel (GPU) → spec mixup → backbone → SED head.
"""

import logging
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from birdclef.classifiers import MultiLabelTopKAccuracy

logger = logging.getLogger(__name__)


class SEDFinetuneModule(pl.LightningModule):
    """SED fine-tuning module with attention-based temporal pooling.

    Uses dual soft cross-entropy loss on clipwise (attention-weighted) and
    framewise-max predictions. Supports spectrogram-domain mixup augmentation.

    Args:
        encoder: Backbone encoder with forward_features() returning (B, C, F, T).
        sed_head: SED attention head returning (clipwise_logits, framewise_logits).
        num_classes: Number of output classes.
        lr: Backbone learning rate.
        head_lr: SED head learning rate (typically higher since head is new).
        weight_decay: AdamW weight decay.
        max_epochs: Maximum training epochs for cosine annealing.
        warmup_epochs: Number of warmup epochs.
        threshold: Classification threshold for binary predictions.
        clipwise_weight: Loss weight for clipwise term.
        framewise_weight: Loss weight for framewise-max term.
        blend_clip: Inference blending weight for clipwise predictions.
        blend_frame: Inference blending weight for framewise-max predictions.
        mixup_prob: Probability of applying mixup augmentation (0 to disable).
        mixup_alpha: Beta distribution parameter for mixup lambda.
        loss_type: Loss function type:
            - "soft_ce": Normalize multi-hot labels to sum-to-1, F.cross_entropy.
            - "ce": Standard F.cross_entropy on argmax class with label_smoothing.
        label_smoothing: Label smoothing factor for "ce" loss type (default: 0.0).
        mel_transform: Optional nn.Module for GPU-side mel spectrogram transform.
            When set, training data is raw waveforms and mel is computed on GPU.
            When None (default), training data is pre-computed mel spectrograms.
        waveform_mixup: When True AND mel_transform is set, mixup applies on raw
            waveforms before mel conversion (v3b behavior). When False (default),
            mixup applies on spectrograms after mel conversion. Only relevant
            when mel_transform is set.
    """

    def __init__(
        self,
        encoder: nn.Module,
        sed_head: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        threshold: float = 0.5,
        clipwise_weight: float = 0.5,
        framewise_weight: float = 0.5,
        blend_clip: float = 0.75,
        blend_frame: float = 0.25,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.5,
        loss_type: str = "soft_ce",
        label_smoothing: float = 0.05,
        mel_transform: nn.Module | None = None,
        waveform_mixup: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "sed_head", "mel_transform"])

        self.encoder = encoder
        self.sed_head = sed_head
        self.num_classes = num_classes
        self.threshold = threshold
        # GPU-side mel transform. When set, training_step receives raw waveforms.
        self.mel_transform = mel_transform

        # --- Metrics (same as LoRAFinetuneModule) ---
        for split in ("train", "val", "test"):
            setattr(
                self,
                f"{split}_cmap",
                torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes, average="macro"
                ),
            )
            setattr(
                self,
                f"{split}_auroc",
                torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="macro"),
            )
            setattr(
                self,
                f"{split}_f1_macro",
                torchmetrics.F1Score(task="multilabel", average="macro", num_labels=num_classes),
            )
            setattr(
                self,
                f"{split}_f1_micro",
                torchmetrics.F1Score(task="multilabel", average="micro", num_labels=num_classes),
            )
            setattr(self, f"{split}_top1", MultiLabelTopKAccuracy(k=1))
            setattr(self, f"{split}_top5", MultiLabelTopKAccuracy(k=5))

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: mel → backbone features → SED head → dual logits.

        Args:
            mel: Input mel spectrogram, shape (B, 1, n_mels, time_frames).

        Returns:
            clipwise_logits: (B, num_classes) attention-weighted temporal sum.
            framewise_logits: (B, num_classes, T) per-frame predictions.
        """
        features = self.encoder.forward_features(mel)
        return self.sed_head(features)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss based on configured loss_type.

        Supports:
        - "soft_ce": Normalize multi-hot to sum-to-1 soft targets, F.cross_entropy.
        - "ce": Standard CE with label_smoothing on argmax class (single-label approx).

        Args:
            logits: (B, num_classes) raw logits.
            labels: (B, num_classes) multi-hot labels (0s and 1s).

        Returns:
            Scalar loss.
        """
        if self.hparams.loss_type == "ce":
            # Standard CE: use argmax of multi-hot as class index
            targets = labels.argmax(dim=-1)
            return F.cross_entropy(logits, targets, label_smoothing=self.hparams.label_smoothing)
        else:
            # soft_ce: normalize multi-hot to soft distribution
            label_sum = labels.sum(dim=-1, keepdim=True).clamp(min=1)
            soft_targets = labels.float() / label_sum
            return F.cross_entropy(logits, soft_targets)

    def _apply_mixup(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation on input tensor (mel spectrogram or raw waveform).

        Shuffles the batch and mixes inputs with a random lambda.
        Labels are unioned (multi-hot OR) to preserve all active species.
        Works on both spectrograms (B, 1, F, T) and raw waveforms (B, N).
        """
        indices = torch.randperm(x.size(0), device=x.device)
        lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
        x = lam * x + (1 - lam) * x[indices]
        labels = torch.clamp(labels + labels[indices], max=1.0)
        return x, labels

    def _blended_probs(
        self, clipwise_logits: torch.Tensor, framewise_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute blended sigmoid probabilities for metrics and inference."""
        framewise_max = framewise_logits.max(dim=-1).values
        return self.hparams.blend_clip * torch.sigmoid(
            clipwise_logits
        ) + self.hparams.blend_frame * torch.sigmoid(framewise_max)

    def training_step(self, batch, batch_idx):
        x, labels = batch

        if self.mel_transform is not None and self.hparams.waveform_mixup:
            # Waveform mixup mode (v3b): mixup on raw waveforms, then mel on GPU.
            # Mel computation must run in float32 and outside autograd graph:
            # - float32: nnAudio STFT magnitudes can overflow float16 range
            # - no_grad: mel is a fixed transform, not a learnable layer
            if random.random() < self.hparams.mixup_prob:
                x, labels = self._apply_mixup(x, labels)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                mel = self.mel_transform(x.float())
        elif self.mel_transform is not None:
            # GPU mel mode: compute mel on GPU first, then spectrogram mixup.
            # Same training behavior as CPU mel mode but avoids DataLoader bottleneck.
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                mel = self.mel_transform(x.float())
            if random.random() < self.hparams.mixup_prob:
                mel, labels = self._apply_mixup(mel, labels)
        else:
            # CPU mel mode (default): input is pre-computed mel, spectrogram mixup
            mel = x
            if random.random() < self.hparams.mixup_prob:
                mel, labels = self._apply_mixup(mel, labels)

        clipwise_logits, framewise_logits = self(mel)
        framewise_max_logits = framewise_logits.max(dim=-1).values

        # Dual soft CE loss
        loss = self.hparams.clipwise_weight * self._compute_loss(
            clipwise_logits, labels
        ) + self.hparams.framewise_weight * self._compute_loss(framewise_max_logits, labels)

        # Blended probabilities for metrics
        probs = self._blended_probs(clipwise_logits, framewise_logits)
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
        for m in (
            self.train_cmap,
            self.train_auroc,
            self.train_f1_macro,
            self.train_f1_micro,
            self.train_top1,
            self.train_top5,
        ):
            m.reset()

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to mel spectrogram, using GPU mel if available."""
        if self.mel_transform is not None:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                return self.mel_transform(x.float())
        return x

    def validation_step(self, batch, batch_idx):
        mel, labels = batch
        mel = self._to_mel(mel)
        clipwise_logits, framewise_logits = self(mel)
        framewise_max_logits = framewise_logits.max(dim=-1).values

        loss = self.hparams.clipwise_weight * self._compute_loss(
            clipwise_logits, labels
        ) + self.hparams.framewise_weight * self._compute_loss(framewise_max_logits, labels)

        probs = self._blended_probs(clipwise_logits, framewise_logits)
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
        for m in (
            self.val_cmap,
            self.val_auroc,
            self.val_f1_macro,
            self.val_f1_micro,
            self.val_top1,
            self.val_top5,
        ):
            m.reset()

    def test_step(self, batch, batch_idx):
        mel, labels = batch
        mel = self._to_mel(mel)
        clipwise_logits, framewise_logits = self(mel)
        framewise_max_logits = framewise_logits.max(dim=-1).values

        loss = self.hparams.clipwise_weight * self._compute_loss(
            clipwise_logits, labels
        ) + self.hparams.framewise_weight * self._compute_loss(framewise_max_logits, labels)

        probs = self._blended_probs(clipwise_logits, framewise_logits)
        preds = (probs > self.threshold).int()

        self.test_cmap.update(probs, labels.int())
        self.test_auroc.update(probs, labels.int())
        self.test_f1_macro.update(preds, labels.int())
        self.test_f1_micro.update(preds, labels.int())
        self.test_top1.update(probs, labels.int())
        self.test_top5.update(probs, labels.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test/cmap", self.test_cmap.compute(), prog_bar=True)
        self.log("test/auroc", self.test_auroc.compute(), prog_bar=True)
        self.log("test/f1_macro", self.test_f1_macro.compute(), prog_bar=True)
        self.log("test/f1_micro", self.test_f1_micro.compute(), prog_bar=False)
        self.log("test/top1_acc", self.test_top1.compute(), prog_bar=True)
        self.log("test/top5_acc", self.test_top5.compute(), prog_bar=True)
        for m in (
            self.test_cmap,
            self.test_auroc,
            self.test_f1_macro,
            self.test_f1_micro,
            self.test_top1,
            self.test_top5,
        ):
            m.reset()

    def configure_optimizers(self):
        """Two param groups: backbone at lower LR, SED head at higher LR."""
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.sed_head.parameters())

        param_groups = [
            {"params": encoder_params, "lr": self.hparams.lr},
            {"params": head_params, "lr": self.hparams.head_lr},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-7,
        )

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
        mel = batch if isinstance(batch, torch.Tensor) else batch[0]
        clipwise_logits, framewise_logits = self(mel)
        probs = self._blended_probs(clipwise_logits, framewise_logits)
        preds = (probs > self.threshold).int()
        return {
            "probabilities": probs,
            "predictions": preds,
            "clipwise_logits": clipwise_logits,
            "framewise_logits": framewise_logits,
        }
