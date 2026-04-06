"""Heads for frozen Perch v2 features.

Three families:
  - LinearHead / MlpHead: take pooled `embedding` (B, 1536) → (B, C) logits.
  - PerchSEDHead: takes `spatial_embedding` (B, 16, 4, 1536) → dict with
    clipwise_logits (B, C) + framewise_logits (B, C, T). Mirrors exp 102's
    SEDHead architecture adapted for Perch's spatial grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pooled-embedding heads (phase 1)
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    """Single linear layer. This is the canonical "linear probe"."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MlpHead(nn.Module):
    """Two-layer MLP with dropout. Marginal cost on top of linear probe."""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SED head on spatial_embedding (phase 2)
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    """Generalized Mean Pooling on the frequency dimension.

    Collapses (B, D, F, T) → (B, D, T) with a learnable power parameter p.
    At p=1 this is average pool; as p→∞ it approaches max pool.
    Copied from exp 102.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool frequency dimension: (B, D, F, T) → (B, D, T)."""
        return x.clamp(min=self.eps).pow(self.p).mean(dim=-2).pow(1.0 / self.p)


class PerchSEDHead(nn.Module):
    """SED head for Perch v2's spatial_embedding.

    Mirrors exp 102's SEDHead (GeM freq pool → projection → dual Conv1d for
    frame logits + attention → clip logits), adapted for Perch's spatial grid:

        Perch spatial_embedding: (B, 16, 4, 1536) = (B, T, F, D)
        EfficientNet features:   (B, D, F, T)

    So we permute to (B, D, F, T) first, then the rest is identical.

    Args:
        embed_dim: Channel dim of spatial_embedding (1536 for Perch v2).
        num_classes: Number of output species (206 for BirdCLEF 2026).
        hidden_dim: Projection bottleneck dim (default: 512).
        dropout: Dropout rate in projection layer (default: 0.5).
        gem_p: Initial GeM power parameter (default: 3.0).
        blend_clip: Weight for clipwise prediction at inference (default: 0.75).
        blend_frame: Weight for framewise max at inference (default: 0.25).
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        num_classes: int = 206,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        gem_p: float = 3.0,
        blend_clip: float = 0.75,
        blend_frame: float = 0.25,
    ):
        super().__init__()
        self.blend_clip = blend_clip
        self.blend_frame = blend_frame

        self.gem = GeM(p=gem_p)
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.frame_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.att_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, spatial: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass from Perch spatial_embedding to dual logits.

        Args:
            spatial: (B, 16, 4, 1536) = (B, T, F, D) from Perch v2.

        Returns:
            dict with:
                clipwise_logits: (B, C) clip-level predictions
                framewise_logits: (B, C, T) per-frame logits (T=16)
        """
        # Permute from Perch (B, T, F, D) → exp 102 convention (B, D, F, T)
        x = spatial.permute(0, 3, 2, 1)  # (B, 1536, 4, 16)

        # ReLU before GeM: spatial_embedding can be negative, GeM needs
        # non-negative input for the power mean to be well-defined.
        x = torch.relu(x)

        # GeM pool frequency: (B, D, F, T) → (B, D, T)
        x = self.gem(x)  # (B, 1536, 16)

        # Project: (B, D, T) → (B, hidden, T)
        x = x.permute(0, 2, 1)   # (B, 16, 1536)
        x = self.project(x)       # (B, 16, hidden)
        x = x.permute(0, 2, 1)   # (B, hidden, 16)

        # Dual branches: frame logits + attention weights
        frame_logits = self.frame_conv(x)  # (B, C, 16)
        att_logits = self.att_conv(x)      # (B, C, 16)

        # Clip-level: tanh-bounded attention weighted sum of frame logits
        att_weights = torch.softmax(torch.tanh(att_logits), dim=-1)  # (B, C, 16)
        clipwise_logits = (att_weights * frame_logits).sum(dim=-1)   # (B, C)

        if not self.training:
            # Blend with framewise max for brief call detection
            frame_max = frame_logits.max(dim=-1).values  # (B, C)
            clipwise_logits = (
                self.blend_clip * clipwise_logits
                + self.blend_frame * frame_max
            )

        return {
            "clipwise_logits": clipwise_logits,
            "framewise_logits": frame_logits,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_head(name: str, embed_dim: int, num_classes: int, **kwargs) -> nn.Module:
    if name == "linear":
        return LinearHead(embed_dim=embed_dim, num_classes=num_classes)
    if name == "mlp":
        return MlpHead(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
    if name == "sed":
        return PerchSEDHead(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown head {name!r}; expected 'linear', 'mlp', or 'sed'")
