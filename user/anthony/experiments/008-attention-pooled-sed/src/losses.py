"""Focal BCE loss supporting both logits and probability inputs.

When gamma=0, reduces to plain BCE. Supports per-token masking for SED losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal binary cross-entropy loss.

    Args:
        gamma: Focal modulating factor. 0 = plain BCE.
        from_logits: If True, input is raw logits. If False, input is probabilities.
    """

    def __init__(self, gamma: float = 2.0, from_logits: bool = True):
        super().__init__()
        self.gamma = gamma
        self.from_logits = from_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            input: Predictions — logits (B, ..., C) or probabilities (B, ..., C)
            target: Targets (B, ..., C) in [0, 1]
            mask: Optional boolean mask (B, T). True = real token, False = padding.
                  Only used for token-level losses with shape (B, T, C).

        Returns:
            Scalar loss
        """
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
            p = torch.sigmoid(input)
        else:
            p = input.clamp(1e-7, 1 - 1e-7)
            bce = F.binary_cross_entropy(p, target, reduction="none")

        if self.gamma > 0:
            pt = torch.where(target > 0.5, p, 1 - p)
            focal_weight = (1 - pt) ** self.gamma
            bce = focal_weight * bce

        if mask is not None:
            # mask: (B, T), bce: (B, T, C)
            bce = bce * mask.unsqueeze(-1)
            # Average over real tokens and classes
            return bce.sum() / mask.sum().clamp(min=1) / target.shape[-1]

        return bce.mean()
