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
from contextlib import nullcontext

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from birdclef.classifiers import MultiLabelTopKAccuracy

logger = logging.getLogger(__name__)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al. 2021).

    Decouples focusing for positive and negative samples. Suppresses easy
    negatives aggressively while preserving positive gradients — critical for
    the 234-class BirdCLEF setting where >99% of labels are negative.

    Args:
        gamma_neg: Focusing parameter for negatives (default: 4).
        gamma_pos: Focusing parameter for positives (default: 0).
        clip: Probability clipping for negatives to prevent gradient explosion (default: 0.05).
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ASL loss.

        Args:
            logits: (B, C) raw logits.
            targets: (B, C) multi-hot labels (0s and 1s, or soft labels).
        """
        probs = torch.sigmoid(logits)

        # Positive and negative parts
        pos_probs = probs
        neg_probs = 1.0 - probs

        # Probability margin clipping for negatives
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        # Basic CE terms
        pos_loss = -targets * torch.log(pos_probs.clamp(min=1e-8))
        neg_loss = -(1.0 - targets) * torch.log(neg_probs.clamp(min=1e-8))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            pos_loss = pos_loss * (1.0 - pos_probs).pow(self.gamma_pos)
        if self.gamma_neg > 0:
            neg_loss = neg_loss * probs.pow(self.gamma_neg)

        return (pos_loss + neg_loss).sum(dim=-1).mean()


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
            - "bce": Binary cross-entropy with logits (native multi-label, sigmoid-based).
            - "asl": Asymmetric Loss (Ridnik et al. 2021) for multi-label classification.
        label_smoothing: Label smoothing factor for "ce" loss type (default: 0.0).
        asl_gamma_neg: ASL focusing for negatives (default: 4.0).
        asl_gamma_pos: ASL focusing for positives (default: 0.0).
        asl_clip: ASL probability clipping for negatives (default: 0.05).
        mel_transform: Optional nn.Module for GPU-side mel spectrogram transform.
            When set, training data is raw waveforms and mel is computed on GPU.
            When None (default), training data is pre-computed mel spectrograms.
        waveform_mixup: When True AND mel_transform is set, mixup applies on raw
            waveforms before mel conversion (v3b behavior). When False (default),
            mixup applies on spectrograms after mel conversion. Only relevant
            when mel_transform is set.
        mixup_per_sample: When True, sample a separate lambda for each sample in
            the batch (ttahara's approach). When False (default), use a single
            scalar lambda for the entire batch.
        scheduler_type: LR scheduler type. "cosine" (default) uses CosineAnnealing
            with linear warmup (per-epoch). "one_cycle" uses OneCycleLR with a
            single param group and per-step updates (ttahara's approach).
        onecycle_div_factor: OneCycleLR initial_lr = max_lr / div_factor (default: 25.0).
        onecycle_final_div_factor: OneCycleLR final_lr = initial_lr / final_div_factor (default: 4.0).
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
        mixup_warmup_epochs: int = 0,
        mixup_theta: float = 1.0,
        mixup_per_sample: bool = False,
        loss_type: str = "soft_ce",
        label_smoothing: float = 0.05,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        mel_transform: nn.Module | None = None,
        waveform_mixup: bool = False,
        waveform_noise: bool = False,
        waveform_time_stretch: bool = False,
        manifold_mixup: bool = False,
        manifold_mixup_prob: float = 0.5,
        scheduler_type: str = "cosine",
        onecycle_div_factor: float = 25.0,
        onecycle_final_div_factor: float = 4.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "sed_head", "mel_transform"])

        self.encoder = encoder
        self.sed_head = sed_head
        self.num_classes = num_classes
        self.threshold = threshold
        # GPU-side mel transform. When set, training_step receives raw waveforms.
        self.mel_transform = mel_transform

        # ASL loss instance (created once, reused)
        if loss_type == "asl":
            self.asl_loss = AsymmetricLoss(
                gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip
            )

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

    def _mel_grad_context(self):
        """Return no_grad context unless mel_transform has learnable params (e.g. PCEN)."""
        if self.mel_transform is not None and any(
            p.requires_grad for p in self.mel_transform.parameters()
        ):
            return nullcontext()
        return torch.no_grad()

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

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        primary_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss based on configured loss_type.

        Supports:
        - "soft_ce": Normalize multi-hot to sum-to-1 soft targets, F.cross_entropy.
        - "ce": Standard CE on primary_label class index (+ sigmoid at inference).
        - "bce": Binary cross-entropy with logits (native multi-label).
        - "asl": Asymmetric Loss (Ridnik et al. 2021) for multi-label classification.

        Args:
            logits: (B, num_classes) raw logits.
            labels: (B, num_classes) multi-hot labels (0s and 1s).
            primary_idx: (B,) integer tensor with primary species index per sample.
                Used by "ce" loss. Ignored by other loss types.

        Returns:
            Scalar loss.
        """
        if self.hparams.loss_type == "asl":
            return self.asl_loss(logits, labels)
        elif self.hparams.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        elif self.hparams.loss_type == "ce":
            # Standard CE on primary label (not argmax of multi-hot)
            targets = primary_idx if primary_idx is not None else labels.argmax(dim=-1)
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

        Shuffles the batch and mixes inputs with a random lambda from
        Beta(alpha, alpha). Lambda is enforced >= 0.5 so the dominant
        sample stays dominant. Labels are weighted-mixed (not OR union).

        If mixup_theta < 1.0, labels >= theta are clipped to 1.0 (ttahara's
        approach: prevents soft labels from diluting strong positive signal).

        If mixup_per_sample is True, each sample in the batch gets its own
        lambda (ttahara's approach). Otherwise, a single lambda is used for
        the entire batch.

        Works on both spectrograms (B, 1, F, T) and raw waveforms (B, N).
        """
        indices = torch.randperm(x.size(0), device=x.device)

        if self.hparams.mixup_per_sample:
            beta_dist = torch.distributions.Beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
            lam = beta_dist.sample(sample_shape=(x.size(0),)).to(x.device)
            lam = torch.maximum(lam, 1 - lam).float()

            if x.dim() == 4:  # spectrogram (B, 1, F, T)
                x_lam = lam[:, None, None, None]
            else:  # waveform (B, N)
                x_lam = lam[:, None]
            label_lam = lam[:, None]

            x = x_lam * x + (1 - x_lam) * x[indices]
            labels = label_lam * labels + (1 - label_lam) * labels[indices]
        else:
            lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
            lam = max(lam, 1.0 - lam)  # Enforce lam >= 0.5
            x = lam * x + (1 - lam) * x[indices]
            labels = lam * labels + (1 - lam) * labels[indices]

        theta = self.hparams.mixup_theta
        if theta < 1.0:
            labels = torch.where(labels >= theta, torch.ones_like(labels), labels)
        return x, labels

    def _apply_manifold_mixup(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup in embedding space (after backbone, before head).

        Same mixing logic as _apply_mixup but operates on backbone feature maps
        (B, C, F', T') instead of input spectrograms. Interpolation in learned
        feature space produces more semantically meaningful augmentations than
        input-space mixing (Verma et al. 2019, Moummad et al. 2024 "Mix2").

        Enabled by manifold_mixup=True in config. When active, training_step
        randomly alternates between input-space and manifold mixup per batch.
        """
        indices = torch.randperm(features.size(0), device=features.device)

        if self.hparams.mixup_per_sample:
            beta_dist = torch.distributions.Beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
            lam = beta_dist.sample(sample_shape=(features.size(0),)).to(features.device)
            lam = torch.maximum(lam, 1 - lam).float()

            # Broadcast lambda over (C, F', T') dims of backbone features
            feat_lam = lam[:, None, None, None]
            label_lam = lam[:, None]

            features = feat_lam * features + (1 - feat_lam) * features[indices]
            labels = label_lam * labels + (1 - label_lam) * labels[indices]
        else:
            lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
            lam = max(lam, 1.0 - lam)
            features = lam * features + (1 - lam) * features[indices]
            labels = lam * labels + (1 - lam) * labels[indices]

        theta = self.hparams.mixup_theta
        if theta < 1.0:
            labels = torch.where(labels >= theta, torch.ones_like(labels), labels)
        return features, labels

    def _blended_probs(
        self, clipwise_logits: torch.Tensor, framewise_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute blended sigmoid probabilities for metrics and inference.

        Handles multi-chunk clipwise_logits (B, N, C) by averaging probs across chunks.
        """
        # Multi-chunk: average probabilities across chunks for metrics
        if clipwise_logits.dim() == 3:
            clip_probs = torch.sigmoid(clipwise_logits).mean(dim=1)  # (B, N, C) → (B, C)
        else:
            clip_probs = torch.sigmoid(clipwise_logits)  # (B, C)

        framewise_max = framewise_logits.max(dim=-1).values
        return self.hparams.blend_clip * clip_probs + self.hparams.blend_frame * torch.sigmoid(
            framewise_max
        )

    def _compute_chunked_loss(
        self,
        clipwise_logits: torch.Tensor,
        framewise_logits: torch.Tensor,
        labels: torch.Tensor,
        primary_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss handling both single-chunk and multi-chunk clipwise outputs.

        When clipwise_logits is (B, C): standard single-prediction loss (v5j behavior).
        When clipwise_logits is (B, N, C): average loss across N chunks. Each chunk
        is supervised with the same label (correct for focal recordings; approximate
        for soundscapes where flanking chunks may have different species).
        """
        if clipwise_logits.dim() == 3:
            # Multi-chunk: (B, N, C) — average loss across chunks
            n_chunks = clipwise_logits.shape[1]
            chunk_losses = []
            for i in range(n_chunks):
                chunk_clip = clipwise_logits[:, i, :]  # (B, C)
                clip_loss = self._compute_loss(chunk_clip, labels, primary_idx)
                chunk_losses.append(clip_loss)
            clipwise_loss = sum(chunk_losses) / n_chunks
        else:
            # Single-chunk: (B, C) — standard loss
            clipwise_loss = self._compute_loss(clipwise_logits, labels, primary_idx)

        # Framewise loss (always on full temporal extent, unchanged)
        framewise_max_logits = framewise_logits.max(dim=-1).values
        framewise_loss = self._compute_loss(framewise_max_logits, labels, primary_idx)

        return (
            self.hparams.clipwise_weight * clipwise_loss
            + self.hparams.framewise_weight * framewise_loss
        )

    def _mixup_active(self) -> bool:
        """Check if mixup should be applied this epoch (respects warmup)."""
        if self.hparams.mixup_prob <= 0:
            return False
        warmup = self.hparams.mixup_warmup_epochs
        if warmup > 0 and self.current_epoch < warmup:
            return False
        return random.random() < self.hparams.mixup_prob

    @staticmethod
    def _apply_pink_noise(waveform: torch.Tensor, snr_range: tuple = (10.0, 30.0)) -> torch.Tensor:
        """Add pink noise (1/f spectrum) to waveform. SNR sampled uniformly per sample."""
        B, T = waveform.shape
        # Generate white noise, apply 1/f filter in frequency domain
        white = torch.randn_like(waveform)
        freqs = torch.fft.rfftfreq(T, device=waveform.device)
        freqs[0] = 1.0  # avoid division by zero
        pink_filter = 1.0 / torch.sqrt(freqs)
        pink = torch.fft.irfft(torch.fft.rfft(white) * pink_filter, n=T)
        # Normalize pink noise to match signal RMS, then scale by SNR
        snr_db = torch.empty(B, 1, device=waveform.device).uniform_(*snr_range)
        snr_linear = 10 ** (snr_db / 20)
        sig_rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        noise_rms = pink.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        pink_scaled = pink * (sig_rms / (noise_rms * snr_linear))
        return waveform + pink_scaled

    @staticmethod
    def _apply_time_stretch(
        waveform: torch.Tensor, rate_range: tuple = (0.8, 1.25)
    ) -> torch.Tensor:
        """Time-stretch waveform by resampling, then crop/pad to original length."""
        B, T = waveform.shape
        rate = torch.empty(1).uniform_(*rate_range).item()
        new_T = int(T * rate)
        # Simple linear interpolation stretch
        stretched = torch.nn.functional.interpolate(
            waveform.unsqueeze(1), size=new_T, mode="linear", align_corners=False
        ).squeeze(1)
        # Crop or pad to original length
        if new_T > T:
            start = torch.randint(0, new_T - T, (1,)).item()
            stretched = stretched[:, start : start + T]
        elif new_T < T:
            pad = T - new_T
            stretched = torch.nn.functional.pad(stretched, (0, pad))
        return stretched

    def training_step(self, batch, batch_idx):
        x, labels, primary_idx = batch

        # Waveform augmentations (before mixup and mel, only when input is raw audio)
        if self.mel_transform is not None:
            if (
                self.hparams.waveform_noise
                and self.current_epoch >= self.hparams.mixup_warmup_epochs
            ):
                x = self._apply_pink_noise(x)
            if (
                self.hparams.waveform_time_stretch
                and self.current_epoch >= self.hparams.mixup_warmup_epochs
            ):
                x = self._apply_time_stretch(x)

        # Decide mixup strategy for this batch. Sample _mixup_active() once to
        # avoid double-sampling the random probability check.
        do_mixup = self._mixup_active()
        use_manifold = (
            do_mixup
            and self.hparams.manifold_mixup
            and random.random() < self.hparams.manifold_mixup_prob
        )

        if not use_manifold:
            # Original input-space mixup paths (unchanged when manifold_mixup=False)
            if self.mel_transform is not None and self.hparams.waveform_mixup:
                # Waveform mixup mode (v3b): mixup on raw waveforms, then mel on GPU.
                if do_mixup:
                    x, labels = self._apply_mixup(x, labels)
                with self._mel_grad_context(), torch.amp.autocast("cuda", enabled=False):
                    mel = self.mel_transform(x.float())
            elif self.mel_transform is not None:
                # GPU mel mode: compute mel on GPU first, then spectrogram mixup.
                with self._mel_grad_context(), torch.amp.autocast("cuda", enabled=False):
                    mel = self.mel_transform(x.float())
                if do_mixup:
                    mel, labels = self._apply_mixup(mel, labels)
            else:
                # CPU mel mode (default): input is pre-computed mel, spectrogram mixup
                mel = x
                if do_mixup:
                    mel, labels = self._apply_mixup(mel, labels)

            clipwise_logits, framewise_logits = self(mel)
        else:
            # Manifold mixup: compute mel (no input-space mixup), run backbone,
            # mix in embedding space, then run head.
            if self.mel_transform is not None:
                with self._mel_grad_context(), torch.amp.autocast("cuda", enabled=False):
                    mel = self.mel_transform(x.float())
            else:
                mel = x

            features = self.encoder.forward_features(mel)
            features, labels = self._apply_manifold_mixup(features, labels)
            clipwise_logits, framewise_logits = self.sed_head(features)

        loss = self._compute_chunked_loss(clipwise_logits, framewise_logits, labels, primary_idx)

        # Blended probabilities for metrics (handles multi-chunk automatically)
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
        mel, labels, primary_idx = batch
        mel = self._to_mel(mel)
        clipwise_logits, framewise_logits = self(mel)

        loss = self._compute_chunked_loss(clipwise_logits, framewise_logits, labels, primary_idx)

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
        mel, labels, primary_idx = batch
        mel = self._to_mel(mel)
        clipwise_logits, framewise_logits = self(mel)

        loss = self._compute_chunked_loss(clipwise_logits, framewise_logits, labels, primary_idx)

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
        """Configure optimizer and LR scheduler.

        Supports two modes:
        - "cosine" (default): Two param groups (encoder/head) with CosineAnnealing + warmup, per-epoch.
        - "one_cycle": Single param group with OneCycleLR, per-step (ttahara's approach).
        """
        if self.hparams.scheduler_type == "one_cycle":
            # Single param group, uniform LR (matching ttahara)
            all_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                epochs=self.hparams.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.hparams.warmup_epochs / self.hparams.max_epochs,
                div_factor=self.hparams.onecycle_div_factor,
                final_div_factor=self.hparams.onecycle_final_div_factor,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        # Default: two param groups with CosineAnnealing + warmup (per-epoch)
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
