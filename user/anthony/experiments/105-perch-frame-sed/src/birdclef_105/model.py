"""Heads that sit on top of frozen Perch v2 pooled embeddings.

All heads take `(B, embed_dim)` input and return `(B, num_classes)` logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


def build_head(name: str, embed_dim: int, num_classes: int, **kwargs) -> nn.Module:
    if name == "linear":
        return LinearHead(embed_dim=embed_dim, num_classes=num_classes)
    if name == "mlp":
        return MlpHead(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown head {name!r}; expected 'linear' or 'mlp'")
