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
        self.norm_f = RMSNorm(config.embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following VisionMamba."""
        # Initialize cls_token and pos_embed
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

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

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension."""
        return self.config.embed_dim
