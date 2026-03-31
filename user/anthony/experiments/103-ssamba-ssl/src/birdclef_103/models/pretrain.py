"""SSAMBA self-supervised pretraining model with dual NCE + MPG losses.

Masked Patch Generation (MPG): MSE reconstruction of masked tokens.
Masked Patch Classification (MPC/NCE): contrastive discrimination of masked tokens.
Combined: L = nce_weight * L_nce + mse_weight * L_mse

Supports two masking modes:
  - "random": uniformly random token masking (original SSAMBA)
  - "span": contiguous span masking (better for Mamba — adjacent 10ms frames
    are nearly identical, so random masking is trivially solvable)

Mel normalization modes:
  - "pcen": Per-Channel Energy Normalization (default) — adaptive per-band AGC,
    suppresses background noise, amplifies quiet calls
  - "amplitude_to_db": 80dB dynamic range clamp + per-clip standardize + [0,1]

Grid jitter: random temporal offset of mel spectrogram before patching.
Preserves sequential order (translation augmentation), Mamba-compatible.
"""

import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.model import SSAMBAModel

logger = logging.getLogger(__name__)


def _nan_check(name, tensor):
    """Log NaN/Inf diagnostics for a tensor."""
    if tensor.isnan().any() or tensor.isinf().any():
        logger.warning(
            f"NaN/Inf in {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
            f"nan={tensor.isnan().sum().item()}, inf={tensor.isinf().sum().item()}, "
            f"min={tensor[tensor.isfinite()].min().item() if tensor.isfinite().any() else 'N/A'}, "
            f"max={tensor[tensor.isfinite()].max().item() if tensor.isfinite().any() else 'N/A'}"
        )
        return True
    return False


class SSAMBAPretrainModel(nn.Module):
    """SSAMBA with dual NCE + MPG for self-supervised pretraining.

    Architecture:
        waveform -> nnAudio mel -> normalize (PCEN or AmplitudeToDB)
        -> grid jitter -> patch embed -> span mask
        -> Mamba backbone -> NCE head + MPG head

    Args:
        config: SSAMBAConfig for the backbone
        mel_spec_layer: nnAudio MelSpectrogram nn.Module
        target_frames: Expected number of mel frames
        mel_norm: "pcen" or "amplitude_to_db"
        mask_ratio: Fraction of tokens to mask (default 0.40)
        mask_mode: "random" or "span" (default "span")
        mask_span_min: Min span length in tokens (default 5 = 50ms)
        mask_span_max: Max span length in tokens (default 15 = 150ms)
        grid_jitter: Max temporal offset in mel frames (default 5 = 50ms, 0=disabled)
        normalize_targets: Per-patch normalize MSE targets (default True)
        loss_fn: "mse" or "l1" for MPG loss (default "mse")
        decoder_depth: 1 (linear) or 2 (MLP with ReLU). Default 2.
        nce_weight: Weight for NCE loss (default 1.0)
        mse_weight: Weight for MSE loss (default 10.0)
        nce_temperature: Temperature for cosine similarity in NCE (default 0.1)
        nce_num_negatives: Number of cross-sample negatives per prediction (default 100)
    """

    def __init__(
        self,
        config: SSAMBAConfig,
        mel_spec_layer: nn.Module,
        target_frames: int = 3000,
        mel_norm: str = "pcen",
        mask_ratio: float = 0.40,
        mask_mode: str = "span",
        mask_span_min: int = 5,
        mask_span_max: int = 15,
        grid_jitter: int = 5,
        normalize_targets: bool = True,
        loss_fn: str = "mse",
        decoder_depth: int = 2,
        nce_weight: float = 1.0,
        mse_weight: float = 10.0,
        nce_temperature: float = 0.1,
        nce_num_negatives: int = 100,
    ):
        super().__init__()
        self.mel_spec = mel_spec_layer
        self.target_frames = target_frames
        self.mel_norm = mel_norm
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.mask_span_min = mask_span_min
        self.mask_span_max = mask_span_max
        self.grid_jitter = grid_jitter
        self.normalize_targets = normalize_targets
        self.loss_fn = loss_fn
        self.decoder_depth = decoder_depth
        self.nce_weight = nce_weight
        self.mse_weight = mse_weight
        self.nce_temperature = nce_temperature
        self.nce_num_negatives = nce_num_negatives

        self.backbone = SSAMBAModel(config)

        embed_dim = config.embed_dim
        fpatch, tpatch = config.fpatch, config.tpatch
        self.patch_dim = fpatch * tpatch

        # Unfold layer to extract raw patch values as targets
        self.unfold = nn.Unfold(
            kernel_size=(fpatch, tpatch),
            stride=(fpatch, tpatch),
        )

        # Learnable mask embedding (replaces masked tokens)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_normal_(self.mask_embed)

        # MPG decoder head (generative — MSE reconstruction)
        if decoder_depth == 1:
            self.gpredlayer = nn.Linear(embed_dim, self.patch_dim)
        else:
            self.gpredlayer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.patch_dim),
            )

        # MPC decoder head (discriminative — NCE contrastive)
        if decoder_depth == 1:
            self.cpredlayer = nn.Linear(embed_dim, self.patch_dim)
        else:
            self.cpredlayer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.patch_dim),
            )

        # PCEN layer (learnable parameters, applied per-channel)
        if mel_norm == "pcen":
            n_mels = fpatch  # 224 for our config
            self.pcen = _PCEN(n_mels=n_mels)

    def _compute_mpg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Per-patch normalization: subtract mean, divide by std."""
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (patches - mean) / std

    @torch.amp.autocast("cuda", enabled=False)
    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute normalized mel spectrogram with configurable normalization.

        Runs in fp32 (autocast disabled) for numerical stability in FFT.
        """
        waveforms = waveforms.float()
        mel = self.mel_spec(waveforms)  # (B, n_mels, T) power spectrogram

        if self.mel_norm == "pcen":
            mel = self.pcen(mel)
        elif self.mel_norm == "amplitude_to_db":
            mel = 10.0 * torch.log10(mel.clamp(min=1e-10))
            mel_max = mel.amax(dim=(-2, -1), keepdim=True)
            mel = torch.clamp(mel, min=mel_max - 80.0)
            mean = mel.mean(dim=(-2, -1), keepdim=True)
            std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            mel = (mel - mean) / std
            mel_min = mel.amin(dim=(-2, -1), keepdim=True)
            mel_max2 = mel.amax(dim=(-2, -1), keepdim=True)
            mel = (mel - mel_min) / (mel_max2 - mel_min + 1e-6)

        T_mel = mel.shape[2]

        # Grid jitter: random temporal offset during training
        if self.grid_jitter > 0 and self.training:
            offset = random.randint(0, self.grid_jitter)
            if offset > 0 and T_mel > offset:
                mel = mel[:, :, offset:]
                T_mel = mel.shape[2]

        if T_mel > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T_mel < self.target_frames:
            pad_len = self.target_frames - T_mel
            mel = F.pad(mel, (0, pad_len), value=0)
        return mel.unsqueeze(1)  # (B, 1, n_mels, T)

    def _generate_span_mask(self, num_real: int, num_total: int, device: torch.device) -> torch.Tensor:
        """Generate contiguous span mask within real (non-padded) tokens.

        Args:
            num_real: Number of real tokens (masking only happens here)
            num_total: Total number of tokens including padding
        """
        mask = torch.zeros(num_total, dtype=torch.bool, device=device)
        target_masked = int(num_real * self.mask_ratio)

        while mask[:num_real].sum() < target_masked:
            span_len = random.randint(self.mask_span_min, self.mask_span_max)
            max_start = num_real - span_len
            if max_start < 0:
                span_len = num_real
                max_start = 0
            start = random.randint(0, max_start)
            mask[start:start + span_len] = True

        # Ensure no masks in padding region
        mask[num_real:] = False
        return mask

    def _generate_random_mask(self, num_real: int, num_total: int, device: torch.device) -> torch.Tensor:
        """Generate uniformly random mask within real tokens."""
        mask_count = int(num_real * self.mask_ratio)
        mask_indices = random.sample(range(num_real), mask_count)
        mask = torch.zeros(num_total, dtype=torch.bool, device=device)
        mask[mask_indices] = True
        return mask

    def forward(self, waveforms: torch.Tensor, real_samples: torch.Tensor | None = None):
        """Forward pass: compute mel, mask, backbone, dual loss.

        Args:
            waveforms: (B, num_samples)
            real_samples: (B,) number of real audio samples per item.
                Used to exclude padded positions from masking/loss.
                If None, all positions are treated as real.

        Returns:
            dict with loss_mpg, loss_nce, nce_accuracy, monitor_mse,
            monitor_l1, viz_mel, viz_mask, viz_pred, viz_target
        """
        B = waveforms.shape[0]
        mel = self.compute_mel(waveforms)  # (B, 1, n_mels, T)
        _nan_check("mel", mel)

        # Extract raw patch values as targets (before embedding)
        raw_patches = self.unfold(mel).transpose(1, 2)  # (B, N, patch_dim)

        # Get patch embeddings from backbone's patch_embed
        patch_embeddings = self.backbone.patch_embed(mel)  # (B, N, D)
        num_patches = patch_embeddings.shape[1]

        # Compute real tokens per sample (for padding-aware masking)
        # Minimum real tokens needed: enough to form at least one mask span
        min_maskable = max(self.mask_span_min, int(1 / self.mask_ratio) + 1)  # e.g. 5 tokens
        if real_samples is not None:
            hop_length = 320  # samples per mel frame
            # real_samples -> real mel frames -> real patches (tpatch=1)
            real_tokens_per_sample = (real_samples.float() / hop_length).long().clamp(max=num_patches)
            min_real = real_tokens_per_sample.min().item()
        else:
            min_real = num_patches

        # Clamp to ensure we have enough tokens to mask
        min_real = max(min_real, min_maskable)

        # Generate mask only within real tokens.
        # Use min_real so the same mask applies to all samples in batch
        # (required because we index with a single mask_indices_t).
        if self.mask_mode == "span":
            mask_bool = self._generate_span_mask(min_real, num_patches, waveforms.device)
        else:
            mask_bool = self._generate_random_mask(min_real, num_patches, waveforms.device)

        mask_indices_t = mask_bool.nonzero(as_tuple=True)[0]
        mask_count = mask_indices_t.shape[0]

        # Apply masking
        masked_embeddings = patch_embeddings.clone()
        mask_embed = self.mask_embed.to(dtype=masked_embeddings.dtype)
        masked_embeddings[:, mask_indices_t, :] = mask_embed.expand(B, mask_count, -1)

        # Add positional embedding
        x = masked_embeddings + self.backbone.pos_embed[:, 1:, :]

        # Insert CLS token
        if self.backbone.cls_token is not None:
            cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)

            if self.backbone.config.use_middle_cls_token:
                cls_pos = num_patches // 2
            else:
                cls_pos = 0

            if cls_pos == 0:
                x = torch.cat([cls_tokens, x], dim=1)
            else:
                x = torch.cat([x[:, :cls_pos, :], cls_tokens, x[:, cls_pos:, :]], dim=1)
        else:
            cls_pos = None

        x = self.backbone.pos_drop(x)

        # Run through Mamba layers
        _nan_check("pre_mamba", x)
        hidden_states, residual = self.backbone.layers(x)
        _nan_check("post_mamba_hidden", hidden_states)
        if residual is not None:
            _nan_check("post_mamba_residual", residual)

        # Final residual + normalization
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.backbone.drop_path(hidden_states)
        x = self.backbone.norm_f(residual.to(dtype=self.backbone.norm_f.weight.dtype))
        _nan_check("post_norm", x)

        # Remove CLS token to get patch representations
        if cls_pos is not None:
            if cls_pos == 0:
                patch_repr = x[:, 1:, :]
            else:
                patch_repr = torch.cat([x[:, :cls_pos, :], x[:, cls_pos + 1:, :]], dim=1)
        else:
            patch_repr = x
        # patch_repr: (B, N, D)

        masked_repr = patch_repr[:, mask_indices_t, :]  # (B, M, D)
        M = masked_repr.shape[1]

        # --- MPG: reconstruction at masked positions ---
        mpg_pred = self.gpredlayer(masked_repr)  # (B, M, patch_dim)
        _nan_check("mpg_pred", mpg_pred)
        mpg_targets = raw_patches[:, mask_indices_t, :]  # (B, M, patch_dim)
        if self.normalize_targets:
            mpg_targets = self._normalize_patches(mpg_targets)
        _nan_check("mpg_targets", mpg_targets)
        loss_mpg = self._compute_mpg_loss(mpg_pred, mpg_targets)
        _nan_check("loss_mpg", loss_mpg)

        # Cross-metrics: always log both MSE and L1
        with torch.no_grad():
            monitor_mse = F.mse_loss(mpg_pred, mpg_targets)
            monitor_l1 = F.l1_loss(mpg_pred, mpg_targets)

        # --- NCE: cross-sample contrastive with shared negative pool ---
        # Sample K negatives ONCE, shared across all predictions (wav2vec 2.0 style).
        # Memory: O(K * D) for the pool + O(BM * K) for logits. Much cheaper than
        # per-prediction negatives which create a (BM, K, D) gather tensor.
        nce_pred = self.cpredlayer(masked_repr)  # (B, M, patch_dim)
        nce_targets = raw_patches[:, mask_indices_t, :].detach()  # (B, M, patch_dim)

        BM = B * M
        K = min(self.nce_num_negatives, BM - 1)
        with torch.amp.autocast("cuda", enabled=False):
            nce_pred_flat = F.normalize(nce_pred.float().reshape(BM, -1), dim=-1)
            nce_targets_flat = F.normalize(nce_targets.float().reshape(BM, -1), dim=-1)

            # Sample K indices as shared negative pool
            neg_indices = torch.randperm(BM, device=waveforms.device)[:K]
            neg_pool = nce_targets_flat[neg_indices]  # (K, D)

            # Positive similarity: (BM, 1)
            pos_sim = (nce_pred_flat * nce_targets_flat).sum(dim=-1, keepdim=True)
            # Negative similarity: (BM, K) via single matmul
            neg_sim = torch.mm(nce_pred_flat, neg_pool.transpose(0, 1))

            # Logits: positive first, then K negatives → (BM, K+1)
            logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.nce_temperature
            labels = torch.zeros(BM, dtype=torch.long, device=waveforms.device)
            loss_nce = F.cross_entropy(logits, labels)

        with torch.no_grad():
            nce_accuracy = (logits.argmax(dim=-1) == 0).float().mean()

        return {
            "loss_mpg": loss_mpg,
            "loss_nce": loss_nce,
            "nce_accuracy": nce_accuracy,
            "monitor_mse": monitor_mse,
            "monitor_l1": monitor_l1,
            # Stash for visualization (first sample only, detached)
            "viz_mel": mel[0:1].detach(),
            "viz_mask": mask_bool.detach(),
            "viz_pred": mpg_pred[0:1].detach(),
            "viz_target": mpg_targets[0:1].detach(),
        }


class _PCEN(nn.Module):
    """Per-Channel Energy Normalization with learnable parameters.

    PCEN(E) = (E / (smooth(E) + bias)^gain + offset)^power - offset^power

    where smooth(E) is an IIR lowpass filter per channel.

    Ref: Wang et al., "Trainable Frontend For Robust and Far-Field Keyword Spotting", 2017
    """

    def __init__(
        self,
        n_mels: int = 224,
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
        s: float = 0.025,
        eps: float = 1e-6,
        trainable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        if trainable:
            self.log_alpha = nn.Parameter(torch.full((n_mels,), _inv_sigmoid(alpha)))
            self.log_delta = nn.Parameter(torch.full((n_mels,), _inv_softplus(delta)))
            self.log_r = nn.Parameter(torch.full((n_mels,), _inv_softplus(r)))
            self.s = nn.Parameter(torch.full((n_mels,), s))
        else:
            self.register_buffer("log_alpha", torch.full((n_mels,), _inv_sigmoid(alpha)))
            self.register_buffer("log_delta", torch.full((n_mels,), _inv_softplus(delta)))
            self.register_buffer("log_r", torch.full((n_mels,), _inv_softplus(r)))
            self.register_buffer("s", torch.full((n_mels,), s))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to mel spectrogram.

        Args:
            mel: (B, n_mels, T) power mel spectrogram

        Returns:
            (B, n_mels, T) PCEN-normalized spectrogram
        """
        alpha = torch.sigmoid(self.log_alpha).unsqueeze(0).unsqueeze(-1)  # (1, F, 1)
        delta = F.softplus(self.log_delta).unsqueeze(0).unsqueeze(-1)
        r = F.softplus(self.log_r).unsqueeze(0).unsqueeze(-1)
        s = self.s.unsqueeze(0).unsqueeze(-1)

        # IIR lowpass: M[t] = alpha * M[t-1] + (1 - alpha) * E[t]
        smooth = self._iir_filter(mel, alpha)

        # PCEN formula
        pcen = (mel / (smooth + delta).pow(r) + s).pow(1.0 / (r + self.eps)) - s.pow(
            1.0 / (r + self.eps)
        )
        return pcen

    def _iir_filter(self, mel: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Causal IIR lowpass filter along time axis.

        M[t] = alpha * M[t-1] + (1 - alpha) * E[t]

        Runs without grad to avoid inplace autograd issues and keep the
        3000-step loop fast. Gradients for alpha flow through the PCEN
        formula itself (mel / smooth^r), not through the IIR unrolling.
        This is standard practice for trainable PCEN.
        """
        with torch.no_grad():
            alpha_sq = alpha.squeeze(-1)  # (1, F)
            one_minus_alpha = 1 - alpha_sq
            smooth = torch.empty_like(mel)
            smooth[:, :, 0] = mel[:, :, 0]
            for t in range(1, mel.shape[2]):
                smooth[:, :, t] = alpha_sq * smooth[:, :, t - 1] + one_minus_alpha * mel[:, :, t]
        return smooth


def _inv_sigmoid(x: float) -> float:
    """Inverse sigmoid for parameter initialization."""
    import math
    return math.log(x / (1.0 - x))


def _inv_softplus(x: float) -> float:
    """Inverse softplus for parameter initialization."""
    import math
    return math.log(math.exp(x) - 1.0)
