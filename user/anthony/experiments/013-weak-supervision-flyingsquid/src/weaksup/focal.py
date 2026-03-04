"""Focal loss functions for linear probes.

Binary focal loss for multi-label species probes and multiclass focal loss
for taxa probes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalBCELoss(nn.Module):
    """Binary focal cross-entropy loss (supports multi-label).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class positive weight tensor of shape (C,), or scalar.
            If None, no class reweighting (equivalent to alpha=0.5).
        gamma: Focusing parameter. Higher = more focus on hard examples.
    """

    def __init__(self, alpha: Tensor | float | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            logits: (N, C) raw logits.
            targets: (N, C) binary targets in {0, 1}.
            sample_weight: (N,) optional per-sample weight in [0, 1].

        Returns:
            Scalar focal BCE loss averaged over samples and classes.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha if isinstance(self.alpha, Tensor) else self.alpha
            alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce
        if sample_weight is not None:
            loss = loss * sample_weight.unsqueeze(1)
        return loss.mean()


class FocalCELoss(nn.Module):
    """Multiclass focal cross-entropy loss.

    FL(p_c) = -alpha_c * (1 - p_c)^gamma * log(p_c)

    Args:
        alpha: Per-class weight tensor of shape (C,). None = uniform.
        gamma: Focusing parameter.
    """

    def __init__(self, alpha: Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (N, C) raw logits.
            targets: (N,) integer class indices.

        Returns:
            Scalar focal CE loss.
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        p = F.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce).mean()
