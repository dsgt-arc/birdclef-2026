"""Sydorskyi 2025 fine-tune LightningModule for BirdCLEF 2026 (B.9).

Reuses exp 032's `DistilledSEDDataModule` for the 2026 data plumbing (focal +
labeled-SS, 5-fold split seed=42, raw-waveform batches) — see
`experiments/034-pretrained-backbone/results.md` § B.8.3 decision. The datamodule
is model-agnostic: it yields `(wavs (B,1,T), labels (B,C), wts, mks)` (or a
5-tuple with per-sample PL loss_weights). All Sydorskyi-specific logic lives here:

    batch (wavs (B,1,T), labels (B,C), ...)
      → squeeze channel → (B, T)
      → (training) RandomFiltering waveform EQ  (Sydorskyi `batch_aug`)
      → XCPretrainWaveClassifier (mel + spec_augment + backbone + AttHead)
      → FocalLossBCE on `clipwise_logits_long`
      → val AUROC on `clipwise_pred_long` (TuckerMacroAUC — same metric as the
        exp 032 e2e gate, so numbers are comparable)

Label smoothing (α=0.05, mass-preserving) applied to TRAINING targets only.
Optimizer/scheduler default to Sydorskyi's EffNetV2-S recipe: AdamW lr=1e-4,
CosineAnnealing → 1e-6, per-step.

Sydorskyi recipe pieces NOT ported here (deferred; LOW-risk per the research):
MixUp sum-clip, 40% PL swap, BG-noise OneOf. exp 032's datamodule supplies its
own focal/SC mixing (and optional PL). The class-balanced sampler (B.7,
`xcpretrain_sampler`) is available as a standalone utility but the datamodule
uses its own MixSamp by default.

Confidence: VERIFIED ports (Phase A.2) wired into a reuse-exp032 loop. 2026-05-28.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from birdclef.losses.focal_bce import FocalLossBCE
from birdclef.preprocessing.random_filtering import RandomFiltering


class XCPretrainFinetuneModule(pl.LightningModule):
    """Fine-tune an XCPretrainWaveClassifier on exp 032 DistilledSEDDataModule batches.

    Args:
        model: an instantiated `XCPretrainWaveClassifier` (or any nn.Module taking
            a (B, T) waveform and returning a dict with keys
            `clipwise_logits_long` (logits) and `clipwise_pred_long` (probs), and
            having a `.backbone` submodule for the freeze option).
        num_classes: number of output classes (BirdCLEF 2026 = 234).
        lr / weight_decay / max_epochs / eta_min: AdamW + CosineAnnealingLR.
        freeze_backbone: if True, freeze `model.backbone` params (Phase D Gate 2 /
            Phase E frozen-backbone fine-tune).
        label_smoothing: mass-preserving smoothing applied to training targets
            (Sydorskyi α=0.05; set 0 to disable).
        use_random_filtering: apply Sydorskyi's RandomFiltering waveform EQ in
            training (his `batch_aug`).
        rf_min_db: RandomFiltering min dB (Sydorskyi -20).
        focal_alpha / focal_gamma: FocalLossBCE params (Sydorskyi 0.25 / 2.0).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 234,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 50,
        eta_min: float = 1e-6,
        freeze_backbone: bool = False,
        label_smoothing: float = 0.05,
        use_random_filtering: bool = True,
        rf_min_db: float = -20.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.label_smoothing = label_smoothing

        self.loss_fn = FocalLossBCE(alpha=focal_alpha, gamma=focal_gamma)
        self.batch_aug = (
            RandomFiltering(min_db=rf_min_db, is_wave=True, normalize_wave=True)
            if use_random_filtering
            else None
        )

        if freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        from birdclef.metrics import TuckerMacroAUC

        self.val_auroc = TuckerMacroAUC(num_labels=num_classes)

    # ---- helpers -----------------------------------------------------------
    def _smooth(self, targets: torch.Tensor) -> torch.Tensor:
        """Mass-preserving label smoothing (Sydorskyi wave_dataset.py:585-602)."""
        a = self.label_smoothing
        if a and a > 0:
            return targets * (1 - a) + a * targets.sum(dim=-1, keepdim=True) / targets.shape[-1]
        return targets

    @staticmethod
    def _waveform(batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack exp 032/033 batch → (wave (B, T), labels (B, C) float)."""
        x = batch[0]
        labels = batch[1].float()
        if x.dim() == 3:  # (B, 1, T) → (B, T); XCPretrain mel expects (B, T)
            x = x.squeeze(1)
        return x, labels

    def _step(self, batch, *, training: bool):
        """Core forward + loss (no logging — unit-testable without a Trainer)."""
        wave, labels = self._waveform(batch)
        if training and self.batch_aug is not None:
            wave = self.batch_aug(wave)
        out = self.model(wave)
        targets = self._smooth(labels) if training else labels
        loss = self.loss_fn(out["clipwise_logits_long"], targets)
        return loss, out, labels

    def forward(self, wave: torch.Tensor) -> dict:
        return self.model(wave)

    # ---- Lightning hooks ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, _, labels = self._step(batch, training=True)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=labels.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out, labels = self._step(batch, training=False)
        self.val_auroc.update(out["clipwise_pred_long"], labels.int())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=labels.shape[0])
        return loss

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8, betas=(0.9, 0.999)
        )
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            total_steps = self.max_epochs * 100  # standalone fallback (unit tests)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(total_steps, 1), eta_min=self.eta_min
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
