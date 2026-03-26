"""Focal BCE loss. Vendored from exp 009."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal binary cross-entropy loss with optional per-class weighting.

    Args:
        gamma: Focal modulating factor. 0 = plain BCE.
        from_logits: If True, input is raw logits. If False, input is probabilities.
        class_weight: Optional per-class weights (C,). E.g. count^(-0.5) for
            inverse square root frequency weighting.
    """

    def __init__(self, gamma: float = 2.0, from_logits: bool = True,
                 class_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.from_logits = from_logits
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            input: Predictions — logits (B, C) or probabilities (B, C)
            target: Targets (B, C) in [0, 1]
            sample_weight: Optional per-sample weights (B,)

        Returns:
            Scalar loss
        """
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
            p = torch.sigmoid(input)
        else:
            p = input.float().clamp(1e-7, 1 - 1e-7)
            target = target.float()
            bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))

        if self.gamma > 0:
            pt = torch.where(target > 0.5, p, 1 - p)
            focal_weight = (1 - pt) ** self.gamma
            bce = focal_weight * bce

        if self.class_weight is not None:
            # class_weight: (C,) → (1, C) to broadcast over batch
            bce = bce * self.class_weight.unsqueeze(0)

        if sample_weight is not None:
            # sample_weight: (B,) → (B, 1) to broadcast over classes
            bce = bce * sample_weight.unsqueeze(-1)

        return bce.mean()
