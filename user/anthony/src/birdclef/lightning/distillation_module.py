"""PyTorch Lightning module for distillation training.

Wraps student model with DistillationLoss and optimizer configuration.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from birdclef.distillation import DistillationLoss


class DistillationLitModule(pl.LightningModule):
    """Lightning module for distillation training.

    Wraps a student model and trains it to match teacher embeddings
    using combined MSE and cosine similarity loss.

    Args:
        student: Student model that transforms static embeddings to contextualized
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for AdamW (default: 0.01)
        mse_weight: Weight for MSE loss component (default: 0.5)
        cosine_weight: Weight for cosine similarity loss component (default: 0.5)
        max_epochs: Maximum epochs for cosine annealing scheduler
        eta_min: Minimum learning rate for cosine annealing (default: 1e-6)
        grad_clip_val: Gradient clipping value (default: 1.0)
    """

    def __init__(
        self,
        student: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
        max_epochs: int = 20,
        eta_min: float = 1e-6,
        grad_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["student"])

        self.student = student
        self.criterion = DistillationLoss(mse_weight, cosine_weight)

    def forward(self, static_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass through student model.

        Args:
            static_embeds: (batch, seq_len, embed_dim) static embeddings

        Returns:
            Contextualized embeddings (batch, seq_len, embed_dim)
        """
        return self.student(static_embeds)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (static_embeds, teacher_embeds)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        static_embeds, teacher_embeds = batch
        student_embeds = self(static_embeds)
        loss_dict = self.criterion(student_embeds, teacher_embeds)

        # Log training metrics
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mse", loss_dict["mse"], on_step=False, on_epoch=True)
        self.log("train/cosine_loss", loss_dict["cosine"], on_step=False, on_epoch=True)
        self.log("train/cosine_sim", loss_dict["cosine_sim"], on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (static_embeds, teacher_embeds)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        static_embeds, teacher_embeds = batch
        student_embeds = self(static_embeds)
        loss_dict = self.criterion(student_embeds, teacher_embeds)

        # Log validation metrics
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", loss_dict["mse"], on_step=False, on_epoch=True)
        self.log("val/cosine_loss", loss_dict["cosine"], on_step=False, on_epoch=True)
        self.log("val/cosine_sim", loss_dict["cosine_sim"], on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and lr_scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip gradients before optimizer step."""
        if self.hparams.grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.hparams.grad_clip_val
            )
