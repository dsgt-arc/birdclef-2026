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

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from birdclef.classifiers import MultiLabelTopKAccuracy
from birdclef.losses.focal_bce import FocalLossBCE
from birdclef.metrics import TuckerMacroAP, TuckerMacroAUC
from birdclef.preprocessing.random_filtering import RandomFiltering

logger = logging.getLogger(__name__)


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
        use_perch_distill: enable the Perch v2 embedding-KD branch (Phase H). Mirrors
            exp 032's DistilledSEDFinetuneModule: project the backbone feature map
            (GAP→Linear→`perch_embed_dim`) and MSE against a frozen Perch v2 ONNX
            teacher. Requires a trainable backbone (freeze_backbone=False).
        perch_onnx_path: path to the Perch v2 ONNX (required when KD on).
        perch_embed_dim: Perch clip-embedding dim (1536).
        alpha_distill: KD MSE weight (exp 032 / Tucker default 1.0; re-tune if it
            swamps the FocalBCE cls term — watch train/dist_loss vs train/cls_loss).
        kd_detach_cls_features: True → Tucker-faithful (backbone grad ONLY from the
            Perch MSE; SED head reads detached features). False → joint (backbone gets
            BOTH the task and the distill gradient — likely safer for an already-strong
            in-domain backbone).
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
        use_perch_distill: bool = False,
        perch_onnx_path: str | None = None,
        perch_embed_dim: int = 1536,
        alpha_distill: float = 1.0,
        kd_detach_cls_features: bool = True,
        threshold: float = 0.5,
        optimizer_mode: str = "uniform",
        backbone_lr_mult: float = 1.0,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.04,
        unfreeze_backbone_epoch: int = 0,
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

        # --- LR-schedule knobs (Phase A) ---------------------------------------
        self.optimizer_mode = optimizer_mode
        self.backbone_lr_mult = backbone_lr_mult
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        if unfreeze_backbone_epoch > 0 and freeze_backbone:
            raise ValueError(
                "freeze_backbone=True and unfreeze_backbone_epoch>0 are mutually exclusive — "
                "A3 gradual-unfreeze owns the freezing (and keeps the backbone in the optimizer)."
            )

        if freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
        # A3 gradual unfreeze: freeze the backbone now; on_train_epoch_start unfreezes it at
        # `unfreeze_backbone_epoch`. Unlike freeze_backbone, the params STAY in the optimizer
        # (configure_optimizers includes frozen params when unfreeze_backbone_epoch>0) so they
        # start updating immediately on unfreeze.
        if unfreeze_backbone_epoch > 0:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        # --- Perch v2 KD branch (Phase H) ---------------------------------------
        # Port of exp 032 DistilledSEDFinetuneModule KD onto the in-domain backbone.
        # DistillHead is discarded at inference (export takes model.* only), so KD
        # changes weights, not the deployed graph — the same export/kernel works.
        self.use_perch_distill = use_perch_distill
        self.alpha_distill = alpha_distill
        self.kd_detach_cls_features = kd_detach_cls_features
        if use_perch_distill:
            if perch_onnx_path is None:
                raise ValueError("use_perch_distill=True requires perch_onnx_path.")
            if freeze_backbone:
                raise ValueError(
                    "use_perch_distill=True with freeze_backbone=True is a no-op — the "
                    "backbone can't learn from the distill signal. Unfreeze the backbone."
                )
            from birdclef.lightning.sed_finetune import DistillHead, PerchTeacher

            backbone_dim = self.model.backbone.feature_info.channels()[-1]
            self.distill_head = DistillHead(backbone_dim, perch_embed_dim)
            self.perch_teacher = PerchTeacher(perch_onnx_path)
            logger.info(
                f"Perch v2 KD enabled: backbone_dim={backbone_dim}, "
                f"perch_embed_dim={perch_embed_dim}, alpha_distill={alpha_distill}, "
                f"detach_cls_features={kd_detach_cls_features}, onnx={perch_onnx_path}"
            )

        # Full metric suite (matches exp 032 distilled_sed_finetune) on BOTH train + val,
        # logged to W&B + TB. Keeps `self.val_auroc` (tests depend on the name).
        # NOTE: this module aggregates CLIP-ONLY (clipwise_pred_long / _logits_long) — no
        # clip+frame blend — so metrics read clipwise_pred_long directly.
        self.threshold = threshold
        for split in ("train", "val"):
            setattr(self, f"{split}_cmap", TuckerMacroAP(num_labels=num_classes))
            setattr(self, f"{split}_auroc", TuckerMacroAUC(num_labels=num_classes))
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

    def _prep(self, batch, *, training: bool):
        """Unpack → (training) RandomFiltering → smoothed targets.

        Returns (wave (B, T), targets (B, C), labels (B, C)). `targets` are the
        (smoothed in training) supervision; `labels` are the raw 0/1 multi-hot
        used for metrics."""
        wave, labels = self._waveform(batch)
        if training and self.batch_aug is not None:
            wave = self.batch_aug(wave)
        targets = self._smooth(labels) if training else labels
        return wave, targets, labels

    def _step(self, batch, *, training: bool):
        """Core forward + loss (no logging — unit-testable without a Trainer)."""
        wave, targets, labels = self._prep(batch, training=training)
        out = self.model(wave)
        loss = self.loss_fn(out["clipwise_logits_long"], targets)
        return loss, out, labels

    def _perch_embed(self, wave: torch.Tensor) -> torch.Tensor:
        """Frozen Perch v2 clip embedding of the (filtered) waveform → (B, perch_embed_dim).

        Crops/pads to exactly 160000 samples (5s @ 32 kHz), matching the teacher's
        training window (exp 032 DistilledSEDFinetuneModule). The teacher sees the
        SAME post-RandomFiltering waveform the student does; mel + SpecAugment happen
        afterward inside the model (so the teacher never sees them — same as exp 032)."""
        w = wave if wave.dim() == 2 else wave.squeeze(1)
        target_n = 160_000
        n = w.shape[1]
        if n > target_n:
            start = (n - target_n) // 2
            w = w[:, start : start + target_n]
        elif n < target_n:
            w = F.pad(w, (0, target_n - n))
        return self.perch_teacher.embed(w)  # PerchTeacher.embed is @torch.no_grad

    def _step_kd(self, batch):
        """KD training path (Perch v2 embedding distillation).

            cls   = FocalLossBCE(clipwise_logits_long, smoothed_targets)
            dist  = MSE(distill_head(backbone_feat), perch_emb)
            total = cls + alpha_distill · dist

        Backbone gradient flows from `dist` always (distill_head→feat→backbone), and
        from `cls` too iff kd_detach_cls_features=False. Returns the sub-losses for
        logging."""
        wave, targets, labels = self._prep(batch, training=True)
        perch_emb = self._perch_embed(wave)  # (B, embed_dim), no grad
        feat = self.model(
            wave, return_cnn_emb=True
        )  # (B, C, H, W); SpecAugment applied (train mode)
        distill_emb = self.distill_head(feat)  # grad → backbone
        distill_loss = F.mse_loss(distill_emb, perch_emb)
        feat_for_head = feat.detach() if self.kd_detach_cls_features else feat
        out = self.model.head(feat_for_head)
        cls_loss = self.loss_fn(out["clipwise_logits_long"], targets)
        loss = cls_loss + self.alpha_distill * distill_loss
        return loss, out, labels, cls_loss, distill_loss

    def _update_metrics(self, split: str, out: dict, labels: torch.Tensor) -> None:
        """Update the `{split}_*` metric accumulators from a forward output (clip-only)."""
        probs = out["clipwise_pred_long"]
        preds = (probs > self.threshold).int()
        li = labels.int()
        getattr(self, f"{split}_cmap").update(probs, li)
        getattr(self, f"{split}_auroc").update(probs, li)
        getattr(self, f"{split}_f1_macro").update(preds, li)
        getattr(self, f"{split}_f1_micro").update(preds, li)
        getattr(self, f"{split}_top1").update(probs, li)
        getattr(self, f"{split}_top5").update(probs, li)

    def _log_reset_metrics(self, split: str) -> None:
        """Log + reset the `{split}_*` metrics at epoch end."""
        self.log(f"{split}/cmap", getattr(self, f"{split}_cmap").compute(), prog_bar=True)
        self.log(f"{split}/auroc", getattr(self, f"{split}_auroc").compute(), prog_bar=True)
        if split == "val":
            self.log("val/n_evaluable", float(self.val_auroc.n_evaluable()))
        self.log(f"{split}/f1_macro", getattr(self, f"{split}_f1_macro").compute())
        self.log(f"{split}/f1_micro", getattr(self, f"{split}_f1_micro").compute())
        self.log(f"{split}/top1_acc", getattr(self, f"{split}_top1").compute())
        self.log(f"{split}/top5_acc", getattr(self, f"{split}_top5").compute())
        for name in ("cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5"):
            getattr(self, f"{split}_{name}").reset()

    def forward(self, wave: torch.Tensor) -> dict:
        return self.model(wave)

    # ---- Lightning hooks ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.use_perch_distill:
            loss, out, labels, cls_loss, distill_loss = self._step_kd(batch)
            bs = labels.shape[0]
            self.log(
                "train/cls_loss",
                cls_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=bs,
            )
            self.log(
                "train/dist_loss",
                distill_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=bs,
            )
        else:
            loss, out, labels = self._step(batch, training=True)
            bs = labels.shape[0]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        # hengck23 overfit proxy: std of the clip logits (inflates as the model overconfident).
        self.log(
            "train/logit_std",
            out["clipwise_logits_long"].detach().std(),
            on_step=True,
            on_epoch=True,
            batch_size=bs,
        )
        self._update_metrics("train", out, labels)
        return loss

    def on_train_epoch_end(self):
        self._log_reset_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, out, labels = self._step(batch, training=False)
        self._update_metrics("val", out, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=labels.shape[0])
        return loss

    def on_validation_epoch_end(self):
        self._log_reset_metrics("val")

    def on_train_epoch_start(self):
        """A3 gradual unfreeze: flip backbone requires_grad on at the configured epoch."""
        if self.unfreeze_backbone_epoch > 0 and self.current_epoch == self.unfreeze_backbone_epoch:
            for p in self.model.backbone.parameters():
                p.requires_grad = True
            logger.info(f"Gradual unfreeze: backbone unfrozen at epoch {self.current_epoch}")

    def _param_groups(self) -> tuple[list, list]:
        """Split params into (backbone, everything-else) by id — KD distill_head lands in 'else'."""
        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        other = [p for p in self.parameters() if id(p) not in backbone_ids]
        return backbone, other

    def configure_optimizers(self):
        # A3 keeps frozen backbone params in the optimizer so they update on unfreeze.
        include_frozen = self.unfreeze_backbone_epoch > 0

        def _keep(ps):
            return ps if include_frozen else [p for p in ps if p.requires_grad]

        if self.optimizer_mode == "discriminative":
            bb, hd = self._param_groups()
            groups = [
                {"params": _keep(hd), "lr": self.lr},
                {"params": _keep(bb), "lr": self.lr * self.backbone_lr_mult},
            ]
            opt = torch.optim.AdamW(
                groups, weight_decay=self.weight_decay, eps=1e-8, betas=(0.9, 0.999)
            )
        else:  # "uniform" — byte-identical to the prior single-group path
            params = _keep(list(self.parameters()))
            opt = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8, betas=(0.9, 0.999)
            )

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            total_steps = self.max_epochs * 100  # standalone fallback (unit tests)
        total_steps = max(total_steps, 1)

        if self.warmup_epochs and self.warmup_epochs > 0:
            steps_per_epoch = max(total_steps // max(self.max_epochs, 1), 1)
            warmup_steps = self.warmup_epochs * steps_per_epoch
            if warmup_steps >= total_steps:
                warmup_steps = 0  # guard: warmup can't exceed the run
            if warmup_steps > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=self.warmup_start_factor, total_iters=warmup_steps
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=total_steps - warmup_steps, eta_min=self.eta_min
                )
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt, [warmup, cosine], milestones=[warmup_steps]
                )
            else:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=total_steps, eta_min=self.eta_min
                )
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=self.eta_min
            )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
