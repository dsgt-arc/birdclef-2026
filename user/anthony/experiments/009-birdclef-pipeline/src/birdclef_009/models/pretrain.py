"""SSAMBA self-supervised pretraining model with MPG.

Masked Patch Generation (MPG): reconstruction of masked tokens.
Supports two masking modes:
  - "random": uniformly random token masking (original SSAMBA)
  - "span": contiguous span masking (HuBERT/wav2vec 2.0 style, better for Mamba)

Span lengths can be fixed or variable (Uniform[min, max]) to prevent
rhythmic artifacts in learned features.

Grid jitter: random temporal offset of mel spectrogram before patching.
Preserves sequential order (translation augmentation), Mamba-compatible.

JiTTER (optional): Block-level temporal shuffling. Disabled by default
because shuffling fights Mamba's sequential state dynamics.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.model import SSAMBAModel


class SSAMBAPretrainModel(nn.Module):
    """SSAMBA with MPG for self-supervised pretraining.

    Architecture:
        waveform -> nnAudio mel -> grid jitter -> patch embed -> span mask
        -> Mamba backbone -> decoder head (reconstruction at masked positions)

    Args:
        config: SSAMBAConfig for the backbone
        mel_spec_layer: nnAudio MelSpectrogram nn.Module
        target_frames: Expected number of mel frames
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        mask_ratio: Fraction of tokens to mask (default 0.78)
        mask_mode: "random" or "span" (default "span")
        mask_span_min: Min span length in tokens (default 5 = 50ms)
        mask_span_max: Max span length in tokens (default 15 = 150ms)
            When min == max, spans are fixed length.
        grid_jitter: Max temporal offset in mel frames (default 5 = 50ms, 0=disabled)
        normalize_targets: Per-patch normalize targets (default True).
            Forces model to predict spectral shape, not loudness.
        loss_fn: "mse" or "l1" (default "mse").
            L1 treats quiet and loud regions equally; better for subtle harmonics.
        decoder_depth: 1 (linear) or 2 (MLP with ReLU). Default 2.
            Linear decoder (SimMIM style) forces stronger encoder features.
        jitter_weight: Weight for JiTTER shuffle loss (default 0, disabled)
        block_size: Number of tokens per shuffle block (default 50 = 500ms)
        shuffle_ratio: Fraction of inner blocks to shuffle (default 0.75)
    """

    def __init__(
        self,
        config: SSAMBAConfig,
        mel_spec_layer: nn.Module,
        target_frames: int = 1000,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        mask_ratio: float = 0.78,
        mask_mode: str = "span",
        mask_span_min: int = 5,
        mask_span_max: int = 15,
        grid_jitter: int = 5,
        normalize_targets: bool = True,
        loss_fn: str = "mse",
        decoder_depth: int = 2,
        jitter_weight: float = 0.0,
        block_size: int = 50,
        shuffle_ratio: float = 0.75,
    ):
        super().__init__()
        self.mel_spec = mel_spec_layer
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.mask_span_min = mask_span_min
        self.mask_span_max = mask_span_max
        self.grid_jitter = grid_jitter
        self.normalize_targets = normalize_targets
        self.loss_fn = loss_fn
        self.decoder_depth = decoder_depth
        self.jitter_weight = jitter_weight
        self.block_size = block_size
        self.shuffle_ratio = shuffle_ratio

        self.backbone = SSAMBAModel(config)

        embed_dim = config.embed_dim
        fpatch, tpatch = config.fpatch, config.tpatch
        self.patch_dim = fpatch * tpatch  # 128 * 1 = 128 for our config

        # Unfold layer to extract raw patch values as targets
        self.unfold = nn.Unfold(
            kernel_size=(fpatch, tpatch),
            stride=(fpatch, tpatch),
        )

        # Learnable mask embedding (replaces masked tokens)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_normal_(self.mask_embed)

        # MPG decoder head
        if decoder_depth == 1:
            # Linear decoder (SimMIM style): forces stronger encoder features
            self.gpredlayer = nn.Linear(embed_dim, self.patch_dim)
        else:
            # 2-layer MLP decoder
            self.gpredlayer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.patch_dim),
            )

        # JiTTER prediction head (only if enabled)
        if jitter_weight > 0:
            if decoder_depth == 1:
                self.jpredlayer = nn.Linear(embed_dim, self.patch_dim)
            else:
                self.jpredlayer = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, self.patch_dim),
                )

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss (MSE or L1)."""
        if self.loss_fn == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Per-patch normalization: subtract mean, divide by std.

        Forces model to predict spectral shape (timbre), not loudness.
        Each 128-dim patch is independently normalized to mean=0, std=1.

        Args:
            patches: (B, N, patch_dim)
        Returns:
            Normalized patches, same shape.
        """
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (patches - mean) / std

    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute normalized mel spectrogram with optional grid jitter."""
        mel = self.mel_spec(waveforms)
        mel = torch.log(mel.clamp(min=1e-10))
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
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
        return mel.unsqueeze(1)

    def _generate_span_mask(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Generate contiguous span mask with variable-length spans.

        Each span length is sampled from Uniform[mask_span_min, mask_span_max].
        Spans are placed at random positions until mask_ratio is reached.
        Overlapping spans create naturally variable-length masked regions.

        Returns:
            Boolean mask tensor of shape (num_patches,), True = masked.
        """
        mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
        target_masked = int(num_patches * self.mask_ratio)

        while mask.sum() < target_masked:
            span_len = random.randint(self.mask_span_min, self.mask_span_max)
            max_start = num_patches - span_len
            if max_start < 0:
                span_len = num_patches
                max_start = 0
            start = random.randint(0, max_start)
            mask[start:start + span_len] = True

        return mask

    def _generate_random_mask(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Generate uniformly random mask (original SSAMBA style).

        Returns:
            Boolean mask tensor of shape (num_patches,), True = masked.
        """
        mask_count = int(num_patches * self.mask_ratio)
        mask_indices = random.sample(range(num_patches), mask_count)
        mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
        mask[mask_indices] = True
        return mask

    def _block_shuffle(self, patch_embeddings: torch.Tensor):
        """Shuffle blocks of patch embeddings (JiTTER-style).

        Keeps first and last blocks as temporal anchors, shuffles inner blocks.

        Args:
            patch_embeddings: (B, N, D)

        Returns:
            shuffled: (B, N, D) with blocks permuted
            shuffle_mask: (N,) boolean, True for positions in shuffled blocks
        """
        B, N, D = patch_embeddings.shape
        num_blocks = N // self.block_size

        # Build block permutation
        perm = list(range(num_blocks))

        if num_blocks > 2:
            inner = list(range(1, num_blocks - 1))
            num_shuffle = int(len(inner) * self.shuffle_ratio)
            if num_shuffle >= 2:
                to_shuffle = random.sample(inner, num_shuffle)
                shuffled_order = to_shuffle.copy()
                random.shuffle(shuffled_order)
                for orig_idx, new_idx in zip(to_shuffle, shuffled_order):
                    perm[orig_idx] = new_idx

        # Build token-level permutation
        token_perm = []
        for block_idx in perm:
            start = block_idx * self.block_size
            token_perm.extend(range(start, start + self.block_size))

        # Handle remainder tokens (keep in place)
        remainder_start = num_blocks * self.block_size
        token_perm.extend(range(remainder_start, N))

        token_perm_t = torch.tensor(token_perm, device=patch_embeddings.device)

        # Apply permutation
        shuffled = patch_embeddings[:, token_perm_t, :]

        # Shuffle mask: True where position changed
        original_order = torch.arange(N, device=patch_embeddings.device)
        shuffle_mask = token_perm_t[:N] != original_order

        return shuffled, shuffle_mask

    def forward(self, waveforms: torch.Tensor):
        """Forward pass: compute mel, mask, backbone, reconstruction loss.

        Args:
            waveforms: (B, num_samples)

        Returns:
            loss_mpg: scalar reconstruction loss at masked positions
            loss_jitter: scalar order-correction loss (0 if disabled)
        """
        B = waveforms.shape[0]
        mel = self.compute_mel(waveforms)  # (B, 1, n_mels, T)

        # Extract raw patch values as targets (before embedding)
        raw_patches = self.unfold(mel).transpose(1, 2)  # (B, N, patch_dim)

        # Get patch embeddings from backbone's patch_embed
        patch_embeddings = self.backbone.patch_embed(mel)  # (B, N, D)
        num_patches = patch_embeddings.shape[1]

        # Block shuffle (if JiTTER enabled — disabled by default)
        if self.jitter_weight > 0:
            shuffled_embeddings, shuffle_mask = self._block_shuffle(patch_embeddings)
        else:
            shuffled_embeddings = patch_embeddings
            shuffle_mask = torch.zeros(num_patches, dtype=torch.bool, device=waveforms.device)

        # Generate mask
        if self.mask_mode == "span":
            mask_bool = self._generate_span_mask(num_patches, waveforms.device)
        else:
            mask_bool = self._generate_random_mask(num_patches, waveforms.device)

        mask_indices_t = mask_bool.nonzero(as_tuple=True)[0]
        mask_count = mask_indices_t.shape[0]

        # Apply masking on (possibly shuffled) embeddings
        masked_embeddings = shuffled_embeddings.clone()
        mask_embed = self.mask_embed.to(dtype=masked_embeddings.dtype)
        masked_embeddings[:, mask_indices_t, :] = mask_embed.expand(B, mask_count, -1)

        # Add positional embedding (ORIGINAL positions)
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
        hidden_states, residual = self.backbone.layers(x)

        # Final residual + normalization
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.backbone.drop_path(hidden_states)
        hidden_states = self.backbone.norm_f(residual.to(dtype=self.backbone.norm_f.weight.dtype))
        x = self.backbone.norm_f(hidden_states)

        # Remove CLS token to get patch representations
        if cls_pos is not None:
            if cls_pos == 0:
                patch_repr = x[:, 1:, :]
            else:
                patch_repr = torch.cat([x[:, :cls_pos, :], x[:, cls_pos + 1:, :]], dim=1)
        else:
            patch_repr = x
        # patch_repr: (B, N, D)

        # --- MPG: reconstruction at masked positions ---
        masked_repr = patch_repr[:, mask_indices_t, :]  # (B, M, D)
        mpg_pred = self.gpredlayer(masked_repr)  # (B, M, patch_dim)
        # Target: original (pre-shuffle) patches at masked positions
        mpg_targets = raw_patches[:, mask_indices_t, :]  # (B, M, patch_dim)
        if self.normalize_targets:
            mpg_targets = self._normalize_patches(mpg_targets)
        loss_mpg = self._compute_loss(mpg_pred, mpg_targets)

        # Cross-metrics: always log both MSE and L1 for fair comparison
        with torch.no_grad():
            monitor_mse = F.mse_loss(mpg_pred, mpg_targets)
            monitor_l1 = F.l1_loss(mpg_pred, mpg_targets)

        # --- JiTTER: order-correction at unmasked+shuffled positions ---
        if self.jitter_weight > 0:
            # Positions that are unmasked AND were shuffled
            jitter_positions = (~mask_bool) & shuffle_mask
            if jitter_positions.any():
                jitter_repr = patch_repr[:, jitter_positions, :]  # (B, J, D)
                jitter_pred = self.jpredlayer(jitter_repr)  # (B, J, patch_dim)
                jitter_targets = raw_patches[:, jitter_positions, :]
                if self.normalize_targets:
                    jitter_targets = self._normalize_patches(jitter_targets)
                loss_jitter = self._compute_loss(jitter_pred, jitter_targets)
            else:
                loss_jitter = torch.tensor(0.0, device=waveforms.device)
        else:
            loss_jitter = torch.tensor(0.0, device=waveforms.device)

        return {
            "loss_mpg": loss_mpg,
            "loss_jitter": loss_jitter,
            "monitor_mse": monitor_mse,
            "monitor_l1": monitor_l1,
            # Stash for visualization (first sample only, detached)
            "viz_mel": mel[0:1].detach(),          # (1, 1, F, T)
            "viz_mask": mask_bool.detach(),          # (N,)
            "viz_pred": mpg_pred[0:1].detach(),      # (1, M, patch_dim)
            "viz_target": mpg_targets[0:1].detach(), # (1, M, patch_dim)
        }
