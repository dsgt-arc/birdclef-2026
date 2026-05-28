"""Distilled SED fine-tuning module (exp 032 — Tucker recipe replication).

Faithful port of `tuckerarrants/bc2026-distilled-sed` cell 14 — kept as a
separate module from `SEDFinetuneModule` to preserve exp 030/031 paths bit-
identical while exp 032 iterates on the new recipe.

Key differences from `SEDFinetuneModule`:
- **Loss**: dual-supervision only — `0.5 · BCE(clip_logits) + 0.5 · BCE(frame_max_logits)`.
  No soft_ce / ce / asl. No multi-context chunked loss. No multi-context heads.
- **Mixup**: applied at the dataset level (FocalDS in exp 032), NOT inside
  training_step. The module receives already-mixed waveforms.
- **Mel pipeline**: GPU-side `mel_transform` is required (Tucker computes mel
  on GPU). Per-sample mel normalization (subtract mean, divide std) applied
  in training_step, matching Tucker's notebook exactly. Optional `spec_augment`
  module applied AFTER per-sample norm.
- **KD branch**: when `use_perch_distill=True`, the SED head receives
  `features.detach()` so backbone gradients come only from the MSE-against-
  Perch distillation loss. This matches Tucker's `BirdSEDModel.forward` design.
- **Inference blend**: validation_step uses `0.5 · σ(clip) + 0.5 · σ(frame_max)`
  per-window in probability space (matches Tucker's `_predict_from_waveforms`).
  Logit-space blending for submission-time fold ensembling lives in
  `submission.py`, not here.
- **Schedule**: LinearLR(2 ep warmup, start_factor=1/25) → CosineAnnealingLR
  (23 ep, eta_min=1e-6) via SequentialLR, **per-step**. Single param group,
  AdamW(lr=5e-4, weight_decay=1e-4). Matches Tucker's notebook cell 14
  schedule exactly.

References:
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 14)

`DistillHead` and `PerchTeacher` are imported from `sed_finetune` rather than
duplicated — they're leaf utility classes (one is a Linear layer, the other an
ONNX session wrapper). The reproducibility-critical math lives in this module.
"""

import logging
from contextlib import nullcontext

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from birdclef.classifiers import MultiLabelTopKAccuracy
from birdclef.lightning.sed_finetune import DistillHead, PerchTeacher

logger = logging.getLogger(__name__)


class DistilledSEDFinetuneModule(pl.LightningModule):
    """SED fine-tuning module for the distilled-SED recipe (Tucker).

    Forward path:
        x (B, N_samples) → mel_transform → per-sample norm → (training only) spec_augment
        → encoder.forward_features → (KD on: detach for cls branch)
        → sed_head → (clip_logits, frame_logits)

    Loss:
        cls = clipwise_weight · BCE(clip_logits, lb) + framewise_weight · BCE(frame_max, lb)
        dist (KD on) = MSE(distill_head(features), perch_emb)
        total = cls + α · dist

    Inference (validation_step, predict_step):
        probs = blend_clip · σ(clip_logits) + blend_frame · σ(frame_max)

    Submission-time multi-fold ensembling is done in logit space in submission.py
    (per dataset+inference audit § 7.3); this module does the per-window blend.

    Args:
        encoder: Backbone with `forward_features(mel) -> (B, C, F, T)`. For
            exp 032: `EfficientNetEncoder(..., use_post_head=True)` for 1280-d.
        sed_head: Returns `(clip_logits, frame_logits)`. For exp 032:
            `AttHeadLogitSpace`.
        num_classes: Output classes (234 for BirdCLEF 2026).
        mel_transform: GPU-side mel transform. REQUIRED. Tucker computes mel on
            GPU; CPU-side pre-computed mel is not supported here.
        spec_augment: Optional. Applied AFTER per-sample mel normalization,
            training only. None disables (default).
        lr: AdamW learning rate. Tucker's default: 5e-4.
        weight_decay: AdamW weight decay. Tucker's default: 1e-4.
        max_epochs: Total training epochs. Tucker's default: 25.
        warmup_epochs: Linear warmup epochs. Tucker's default: 2.
        warmup_start_factor: LinearLR start factor (so initial_lr = lr * factor).
            Tucker's default: 1/25.
        eta_min: Cosine final LR. Tucker's default: 1e-6.
        clipwise_weight, framewise_weight: Loss weights (default 0.5/0.5).
        blend_clip, blend_frame: Inference blend weights (default 0.5/0.5).
        threshold: Binary prediction threshold for metric updates (default 0.5).
        use_perch_distill: Enable Perch v2 KD branch (default False).
        perch_onnx_path: Path to Perch v2 ONNX file. Required when KD on.
        perch_embed_dim: Perch embedding dim (default 1536).
        alpha_distill: KD MSE weight. Tucker's default: 1.0.
        per_sample_mel_norm: Apply per-sample (mean, std) normalization on mel
            before SpecAugment (default True — Tucker does this in cell 14).
    """

    def __init__(
        self,
        encoder: nn.Module,
        sed_head: nn.Module,
        num_classes: int,
        mel_transform: nn.Module,
        spec_augment: nn.Module | None = None,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 25,
        warmup_epochs: int = 2,
        warmup_start_factor: float = 1.0 / 25,
        eta_min: float = 1e-6,
        clipwise_weight: float = 0.5,
        framewise_weight: float = 0.5,
        blend_clip: float = 0.5,
        blend_frame: float = 0.5,
        threshold: float = 0.5,
        use_perch_distill: bool = False,
        perch_onnx_path: str | None = None,
        perch_embed_dim: int = 1536,
        alpha_distill: float = 1.0,
        per_sample_mel_norm: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "sed_head", "mel_transform", "spec_augment"])

        if mel_transform is None:
            raise ValueError(
                "DistilledSEDFinetuneModule requires a GPU-side mel_transform. "
                "Tucker's recipe computes mel on GPU; pre-computed mel inputs "
                "are not supported in this module."
            )

        self.encoder = encoder
        self.sed_head = sed_head
        self.num_classes = num_classes
        self.mel_transform = mel_transform
        self.spec_augment = spec_augment
        self.threshold = threshold

        # --- KD branch ---
        if use_perch_distill:
            if perch_onnx_path is None:
                raise ValueError("use_perch_distill=True requires perch_onnx_path.")
            backbone_dim = getattr(encoder, "embed_dim", None)
            if backbone_dim is None:
                raise ValueError(
                    "use_perch_distill=True requires encoder.embed_dim attribute "
                    "(present on EfficientNetEncoder)."
                )
            self.distill_head = DistillHead(backbone_dim, perch_embed_dim)
            self.perch_teacher = PerchTeacher(perch_onnx_path)
            logger.info(
                f"Perch v2 KD enabled: backbone_dim={backbone_dim}, "
                f"perch_embed_dim={perch_embed_dim}, alpha_distill={alpha_distill}, "
                f"onnx={perch_onnx_path}"
            )

        # --- Metrics ---
        # Tucker-style: skip 0-positive classes from macro AUROC / AP.
        # Replaces torchmetrics.MultilabelAUROC / MultilabelAveragePrecision
        # which both return 0.0 (not NaN) for 0-positive classes and drag the
        # macro average down. On exp 032's val fold (168 SS windows, 205 of
        # 234 classes with 0 positives) this drops val/auroc from 0.89 → 0.11.
        # See `src/birdclef/metrics/tucker_macro.py` + `scripts/diag_val_auroc.py`.
        from birdclef.metrics import TuckerMacroAP, TuckerMacroAUC

        for split in ("train", "val", "test"):
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

    # ------------------------------------------------------------------
    # Forward + helpers
    # ------------------------------------------------------------------

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward from mel → (clip_logits, frame_logits). Used by validation +
        predict; training_step branches around this to support the KD detach."""
        features = self.encoder.forward_features(mel)
        return self.sed_head(features)

    def _mel_grad_context(self):
        """No-grad context unless mel_transform has learnable params."""
        if any(p.requires_grad for p in self.mel_transform.parameters()):
            return nullcontext()
        return torch.no_grad()

    def _compute_mel(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Mel transform → optional per-sample norm → (training only) SpecAugment."""
        with self._mel_grad_context(), torch.amp.autocast("cuda", enabled=False):
            mel = self.mel_transform(x.float())
        if self.hparams.per_sample_mel_norm:
            # (B, 1, F, T) → per-sample (mean, std) over (1, F, T)
            mean = mel.mean(dim=(1, 2, 3), keepdim=True)
            std = mel.std(dim=(1, 2, 3), keepdim=True)
            mel = (mel - mean) / (std + 1e-6)
        if training and self.spec_augment is not None:
            mel = self.spec_augment(mel)
        return mel

    def _dual_supervision_loss(
        self,
        clip_logits: torch.Tensor,
        frame_logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`clipwise_weight · BCE(clip, lb) + framewise_weight · BCE(frame_max, lb)`.

        Tucker's notebook cell 14 (verbatim):
            bce_clip  = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
            bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
            bce = 0.5 * bce_clip + 0.5 * bce_frame
            ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
            cls_loss = (ps * sw).mean()

        Per the 2026-05-23 dataset+inference audit § 4: `wt`, `mk` are always
        all-ones in the public notebook (their class-weighting / class-masking
        plumbing is unused), and `sw` is 1.0 for focal/sc and 0.0 for
        `focal_missing`. With wt=mk=ones, the formula reduces to plain mean
        BCE — which is what `F.binary_cross_entropy_with_logits(..., reduction='mean')`
        gives us here.

        **Per-sample loss_weight** (exp 033 — own-teacher PL, Antoine 1:1 protocol):
        When `loss_weight=None`, the formula simplifies to the original mean BCE
        (backward-compatible with exp 032). When `loss_weight` is a (B,) tensor,
        each sample's per-class BCE is averaged over classes, scaled by its
        weight, then averaged over the batch — matching the exp 025
        `sed_finetune.py:388-393` per-sample reduction pattern.

        **Known deviation**: we do not zero `focal_missing` samples via `sw`.
        Our pipeline guarantees no missing cache files (FocalDS load returns
        a real waveform or the row is filtered upstream), so this is a no-op
        in practice. If we ever pipe in `focal_missing`-style rows, this loss
        would dilute by the missing-row count (Tucker's `(ps * sw).mean()`
        keeps the divisor at `B` — same behavior with sw=0 contribution).

        Returns (loss, frame_max_logits) — frame_max is reused by the inference
        blend in metrics.
        """
        frame_max = frame_logits.max(dim=-1).values  # (B, num_classes)
        if loss_weight is None:
            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, labels.float())
            bce_frame = F.binary_cross_entropy_with_logits(frame_max, labels.float())
        else:
            # Per-sample weighted BCE: mean over class dim, scale by weight, mean over batch
            bce_clip_ps = F.binary_cross_entropy_with_logits(
                clip_logits, labels.float(), reduction="none"
            ).mean(dim=1)  # (B,)
            bce_frame_ps = F.binary_cross_entropy_with_logits(
                frame_max, labels.float(), reduction="none"
            ).mean(dim=1)  # (B,)
            bce_clip = (bce_clip_ps * loss_weight).mean()
            bce_frame = (bce_frame_ps * loss_weight).mean()
        loss = self.hparams.clipwise_weight * bce_clip + self.hparams.framewise_weight * bce_frame
        return loss, frame_max

    def _blended_probs(
        self, clip_logits: torch.Tensor, frame_max_logits: torch.Tensor
    ) -> torch.Tensor:
        """`blend_clip · σ(clip) + blend_frame · σ(frame_max)` (Tucker cell 14).
        Probability-space blend for offline metrics. Submission-side multi-fold
        blending happens in logit space in `submission.py`."""
        return self.hparams.blend_clip * torch.sigmoid(
            clip_logits
        ) + self.hparams.blend_frame * torch.sigmoid(frame_max_logits)

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        # Accept variable batch shapes:
        #   - 2-tuple (x, labels): unit tests + fast-dev-run minimal
        #   - 4-tuple (x, labels, wts, mks): Tucker training (exp 032 default
        #     from _collate_drop_extras — wts/mks are (B, num_classes) per-class
        #     and unused per audit § 4)
        #   - 5-tuple (x, labels, wts, mks, loss_weight): PL training (exp 033
        #     — loss_weight is (B,) per-sample)
        # loss_weight is detected by ndim==1 to disambiguate from Tucker's
        # (B, num_classes) `mks` tensor that lives at the same batch[3] index.
        x = batch[0]
        labels = batch[1]
        loss_weight: torch.Tensor | None = None
        if len(batch) > 4:
            lw = batch[4]
            if isinstance(lw, torch.Tensor) and lw.ndim == 1:
                loss_weight = lw.to(x.device).float()
            elif isinstance(lw, int | float):
                loss_weight = torch.full(
                    (x.size(0),), float(lw), device=x.device, dtype=torch.float32
                )

        do_kd = self.hparams.use_perch_distill

        # --- 1. Optional teacher forward on raw waveform (before mel) ---
        perch_emb = None
        if do_kd:
            # Tucker crops to exactly 160000 samples (5s @ 32kHz) for the teacher.
            wav_5s = x
            if x.dim() > 2:  # (B, 1, T) → (B, T)
                wav_5s = wav_5s.squeeze(1)
            N = wav_5s.shape[1]
            target_N = 160_000
            if N > target_N:
                start = (N - target_N) // 2
                wav_5s = wav_5s[:, start : start + target_N]
            elif N < target_N:
                wav_5s = F.pad(wav_5s, (0, target_N - N))
            with torch.no_grad():
                perch_emb = self.perch_teacher.embed(wav_5s)

        # --- 2. Mel + per-sample norm + SpecAugment ---
        mel = self._compute_mel(x, training=True)

        # --- 3. Forward with KD-aware gradient branching ---
        if do_kd:
            features = self.encoder.forward_features(mel)
            distill_emb = self.distill_head(features)  # grad → backbone
            distill_loss = F.mse_loss(distill_emb, perch_emb)
            features_for_head = features.detach()  # cls branch: no backbone grad
            clip_logits, frame_logits = self.sed_head(features_for_head)
        else:
            clip_logits, frame_logits = self(mel)
            distill_loss = None

        # --- 4. Loss ---
        # KD MSE loss (when enabled) is NOT per-sample weighted — it's a
        # feature-space alignment signal between student backbone and
        # frozen Perch teacher, sample-confidence-independent. Only the
        # task BCE branch gets per-sample loss_weight.
        cls_loss, frame_max = self._dual_supervision_loss(
            clip_logits, frame_logits, labels, loss_weight=loss_weight
        )
        if distill_loss is not None:
            loss = cls_loss + self.hparams.alpha_distill * distill_loss
            self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/dist_loss", distill_loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            loss = cls_loss

        # --- 5. Metrics ---
        probs = self._blended_probs(clip_logits, frame_max)
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        labels = batch[1]

        mel = self._compute_mel(x, training=False)
        clip_logits, frame_logits = self(mel)

        cls_loss, frame_max = self._dual_supervision_loss(clip_logits, frame_logits, labels)
        probs = self._blended_probs(clip_logits, frame_max)
        preds = (probs > self.threshold).int()

        self.val_cmap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())
        self.val_f1_macro.update(preds, labels.int())
        self.val_f1_micro.update(preds, labels.int())
        self.val_top1.update(probs, labels.int())
        self.val_top5.update(probs, labels.int())
        self.log(
            "val/loss",
            cls_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return cls_loss

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        # Transparency: how many of num_classes classes had ≥1 positive in this
        # val pool. With Tucker's small SS-only val (168 windows), this is
        # ~30-50 / 234 — the rest are skipped by TuckerMacroAUC/AP.
        self.log("val/n_evaluable", float(self.val_auroc.n_evaluable()), prog_bar=False)
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
        x = batch[0]
        labels = batch[1]
        mel = self._compute_mel(x, training=False)
        clip_logits, frame_logits = self(mel)
        cls_loss, frame_max = self._dual_supervision_loss(clip_logits, frame_logits, labels)
        probs = self._blended_probs(clip_logits, frame_max)
        preds = (probs > self.threshold).int()

        self.test_cmap.update(probs, labels.int())
        self.test_auroc.update(probs, labels.int())
        self.test_f1_macro.update(preds, labels.int())
        self.test_f1_micro.update(preds, labels.int())
        self.test_top1.update(probs, labels.int())
        self.test_top5.update(probs, labels.int())
        self.log("test/loss", cls_loss, on_step=False, on_epoch=True)
        return cls_loss

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

    def predict_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        mel = self._compute_mel(x, training=False)
        clip_logits, frame_logits = self(mel)
        frame_max = frame_logits.max(dim=-1).values
        probs = self._blended_probs(clip_logits, frame_max)
        preds = (probs > self.threshold).int()
        return {
            "probabilities": probs,
            "predictions": preds,
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "frame_max": frame_max,
        }

    # ------------------------------------------------------------------
    # Optimizer + schedule (Tucker cell 14 verbatim)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW (single param group) + LinearLR warmup → CosineAnnealingLR via
        SequentialLR, per-step. Matches Tucker's notebook cell 14:

            opt = AdamW(m.parameters(), lr=LR, weight_decay=WD)
            warmup_sched = LinearLR(opt, start_factor=1/25, end_factor=1.0,
                                     total_iters=warmup_steps)
            cosine_sched = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                                              eta_min=1e-6)
            sch = SequentialLR(opt, [warmup_sched, cosine_sched],
                               milestones=[warmup_steps])
        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        warmup_steps = steps_per_epoch * self.hparams.warmup_epochs
        total_steps = steps_per_epoch * self.hparams.max_epochs
        # Guard: total_steps must exceed warmup_steps so CosineAnnealingLR has
        # nonzero T_max. In fast-dev-run / very-short-schedule edge cases, fall
        # back to no warmup.
        if warmup_steps >= total_steps:
            warmup_steps = 0

        if warmup_steps == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps, 1), eta_min=self.hparams.eta_min
            )
        else:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.hparams.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.hparams.eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
