"""SSAMBA model implementation using HuggingFace transformers."""

import torch
import torch.nn as nn
from typing import Optional

from .config import SSAMBAConfig
from .layers import RMSNorm, AudioPatchEmbed, DropPath
from .mamba import BidirectionalMambaStack


class SSAMBAModel(nn.Module):
    """Self-Supervised Audio Mamba model.

    Pure PyTorch implementation using HuggingFace's MambaMixer.
    Compatible with pretrained SSAMBA weights.
    """

    def __init__(self, config: SSAMBAConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = AudioPatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )

        # CLS token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        else:
            self.cls_token = None

        # Positional embedding (includes CLS token position)
        num_positions = self.patch_embed.num_patches
        if config.use_cls_token:
            num_positions += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, config.embed_dim))

        # Position dropout
        self.pos_drop = nn.Dropout(p=0.0)

        # Drop path (for training regularization)
        self.drop_path = DropPath(config.drop_path_rate)

        # Bidirectional Mamba layers
        self.layers = BidirectionalMambaStack(config)

        # Final normalization
        if config.rms_norm:
            self.norm_f = RMSNorm(config.embed_dim)
        else:
            self.norm_f = nn.LayerNorm(config.embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following VisionMamba."""
        # Initialize cls_token and pos_embed
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def set_drop_path_rate(self, drop_path_rate: float):
        """Set stochastic depth rate across all Mamba blocks and the final residual.

        Distributes linearly-increasing drop rates across layers (standard
        stochastic depth schedule), matching the VisionMamba / timm convention.
        Also updates the model-level drop_path used in the final residual.

        Args:
            drop_path_rate: Maximum drop path rate (applied to the last layer).
                            Earlier layers get proportionally smaller rates.
        """
        depth = len(self.layers.layers)
        dpr = [drop_path_rate * i / (depth - 1) for i in range(depth)] if depth > 1 else [drop_path_rate]
        for i, block in enumerate(self.layers.layers):
            block.drop_path = DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        print(f"Stochastic depth: linearly increasing to {drop_path_rate} across {depth} layers")

    def forward_features(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Extract features from spectrogram.

        Args:
            x: Input spectrogram (batch, 1, freq_bins, time_frames)
            return_all_tokens: If True, return all token embeddings

        Returns:
            If return_all_tokens: (batch, num_tokens, embed_dim)
            Else: (batch, embed_dim) - averaged token embeddings
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)
        num_patches = x.shape[1]

        # Add CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            if self.config.use_middle_cls_token:
                # Insert CLS token in the middle (SSAMBA default)
                token_position = num_patches // 2
                x = torch.cat([x[:, :token_position, :], cls_tokens, x[:, token_position:, :]], dim=1)
                self._cls_token_position = token_position
            else:
                # Insert CLS token at the beginning
                x = torch.cat([cls_tokens, x], dim=1)
                self._cls_token_position = 0

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Bidirectional Mamba layers
        hidden_states, residual = self.layers(x)

        # Final residual connection + normalization (matching VisionMamba)
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        # NOTE: The original SSAMBA finetuningavgtok applies norm_f a second time
        # here (both_models.py line 391). This is arguably a bug in the original
        # code — norm_f is already applied in the residual path above — but since
        # the published 96% result was obtained with this double-norm behaviour,
        # we replicate it for finetuning parity.  If training from scratch, consider
        # removing this redundant call.
        x = self.norm_f(hidden_states)

        if return_all_tokens:
            return x

        # Average all tokens except CLS (following SSAMBA's approach)
        if self.cls_token is not None:
            if self.config.use_middle_cls_token:
                # CLS token is in the middle - exclude it and average the rest
                cls_pos = self._cls_token_position
                # Concatenate tokens before and after CLS
                tokens_before = x[:, :cls_pos, :]
                tokens_after = x[:, cls_pos + 1:, :]
                x = torch.cat([tokens_before, tokens_after], dim=1).mean(dim=1)
            else:
                # CLS token at beginning - skip first token
                x = x[:, 1:, :].mean(dim=1)
        else:
            x = x.mean(dim=1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram (batch, time_frames, freq_bins)
               Note: This matches SSAMBA's expected input format

        Returns:
            Embeddings (batch, embed_dim)
        """
        # Reshape input to match expected format
        # Input: (B, T, F) -> (B, 1, F, T)
        x = x.unsqueeze(1).transpose(2, 3)

        return self.forward_features(x)

    def get_embeddings(
        self,
        x: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Extract embeddings from spectrogram.

        Args:
            x: Input spectrogram (batch, time_frames, freq_bins)
            normalize: If True, L2-normalize embeddings

        Returns:
            Embeddings (batch, embed_dim)
        """
        embeddings = self.forward(x)

        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def adapt_for_finetuning(
        self,
        finetune_img_size: tuple,
        fstride: int,
        tstride: int,
    ):
        """Adapt model for finetuning with different input size and/or overlapping patches.

        Matches the original SSAMBA finetuning procedure (both_models.py lines 222-242):
        1. Replace patch_embed.proj with new stride if different from patch_size
        2. Interpolate positional embeddings to match new patch grid

        Args:
            finetune_img_size: (freq_bins, time_frames) for finetuning input
            fstride: Frequency stride for patch embedding
            tstride: Time stride for patch embedding
        """
        # TODO: adapt_for_finetuning assumes scalar patch_size. Update when
        # rectangular patches need finetuning support (not needed for exp 005).
        patch_size = self.config.patch_size
        embed_dim = self.config.embed_dim

        # Compute pretraining patch grid dimensions
        p_f_dim = self.patch_embed.f_dim
        p_t_dim = self.patch_embed.t_dim
        p_num_patches = p_f_dim * p_t_dim

        # Compute finetuning patch grid dimensions
        f_dim = (finetune_img_size[0] - patch_size) // fstride + 1
        t_dim = (finetune_img_size[1] - patch_size) // tstride + 1
        num_patches = f_dim * t_dim

        print(f"adapt_for_finetuning: pretrain grid ({p_f_dim}x{p_t_dim}={p_num_patches}) "
              f"-> finetune grid ({f_dim}x{t_dim}={num_patches})")

        # Replace patch embedding projection if stride differs
        if fstride != patch_size or tstride != patch_size:
            new_proj = torch.nn.Conv2d(
                self.config.in_chans, embed_dim,
                kernel_size=patch_size,
                stride=(fstride, tstride),
            )
            # Copy weights from pretrained projection (sum across input channels if needed)
            old_weight = self.patch_embed.proj.weight.data
            if old_weight.shape[1] != self.config.in_chans:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(old_weight, dim=1).unsqueeze(1)
                )
            else:
                new_proj.weight = torch.nn.Parameter(old_weight.clone())
            new_proj.bias = torch.nn.Parameter(self.patch_embed.proj.bias.data.clone())
            self.patch_embed.proj = new_proj

        # Update patch embed metadata
        self.patch_embed.img_size = finetune_img_size
        self.patch_embed.fstride = fstride
        self.patch_embed.tstride = tstride
        self.patch_embed.f_dim = f_dim
        self.patch_embed.t_dim = t_dim
        self.patch_embed.num_patches = num_patches

        # Interpolate positional embeddings (matching original SSAMBA: both_models.py:230-242)
        cls_token_num = 1 if self.config.use_cls_token else 0

        # Separate CLS token pos_embed from patch pos_embed
        old_pos_embed = self.pos_embed.data
        cls_pos_embed = old_pos_embed[:, :cls_token_num, :]
        patch_pos_embed = old_pos_embed[:, cls_token_num:, :]

        # Reshape patch pos_embed to 2D grid: (1, embed_dim, p_f_dim, p_t_dim)
        patch_pos_embed = patch_pos_embed.reshape(
            1, p_num_patches, embed_dim
        ).transpose(1, 2).reshape(1, embed_dim, p_f_dim, p_t_dim)

        # Interpolate to new grid size using bilinear interpolation
        if t_dim <= p_t_dim:
            # Cut from center (matching original)
            t_start = p_t_dim // 2 - t_dim // 2
            patch_pos_embed = patch_pos_embed[:, :, :, t_start:t_start + t_dim]
        else:
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed, size=(p_f_dim, t_dim), mode='bilinear', align_corners=False
            )

        if f_dim <= p_f_dim:
            # Cut from center
            f_start = p_f_dim // 2 - f_dim // 2
            patch_pos_embed = patch_pos_embed[:, :, f_start:f_start + f_dim, :]
        else:
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed, size=(f_dim, t_dim), mode='bilinear', align_corners=False
            )

        # Reshape back to (1, num_patches, embed_dim)
        patch_pos_embed = patch_pos_embed.reshape(1, embed_dim, num_patches).transpose(1, 2)

        # Reassemble with CLS token
        new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
        self.pos_embed = torch.nn.Parameter(new_pos_embed)

        print(f"pos_embed shape: {self.pos_embed.shape} "
              f"(expected: (1, {num_patches + cls_token_num}, {embed_dim}))")

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension."""
        return self.config.embed_dim
