"""PyTorch Lightning module for linear probe evaluation of SSL representations."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class LinProbeLightningModule(pl.LightningModule):
    """Linear probe on frozen SSAMBA backbone.

    Trains only the classification head. Backbone remains frozen throughout.

    Args:
        model: SSAMBALinearProbeClassifier (backbone frozen, head trainable)
        num_classes: Number of species classes
        lr: Learning rate for the head
        weight_decay: Weight decay (0 for linear probe)
        max_epochs: For cosine annealing
        warmup_epochs: Linear warmup epochs
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 30,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        # Attention head outputs probabilities (sigmoided), not logits
        self.criterion = nn.BCELoss()

        # Metrics
        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro",
        )
        self.train_auroc = torchmetrics.AUROC(
            task="multilabel", num_labels=num_classes, average="macro",
        )
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro",
        )
        self.val_auroc = torchmetrics.AUROC(
            task="multilabel", num_labels=num_classes, average="macro",
        )

    def training_step(self, batch, batch_idx):
        waveforms, labels, real_samples = batch
        probs = self.model(waveforms, real_samples=real_samples)
        loss = self.criterion(probs, labels)

        self.train_cmap.update(probs, labels.int())
        self.train_auroc.update(probs, labels.int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.log("train/auroc", self.train_auroc.compute())
        self.train_cmap.reset()
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        waveforms, labels, real_samples = batch
        probs = self.model(waveforms, real_samples=real_samples)
        loss = self.criterion(probs, labels)

        self.val_cmap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.val_cmap.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        # Only train head parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6,
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
