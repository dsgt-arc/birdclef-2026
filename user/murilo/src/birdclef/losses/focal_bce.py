"""Sydorskyi 2025 FocalLossBCE — direct port.

Source: vendor/sydorskyi-2025/code_base/losses/focal_loss.py:28-54 @ 053594f

This is the loss selected by Sydorskyi's `loss_type="baseline"` + `use_bce_focal_loss=True`
config path (cls_forward.py:100-106). It is applied to the head's
`clipwise_logits_long` only — there is NO framewise-max auxiliary term in the
"baseline" loss type (that's the separate `BCEFocal2WayLoss`, unused in the
selected configs).

Composition (verified — focal_loss.py:45-54):

    loss = bce_weight * BCEWithLogitsLoss(reduction)(logits, targets)
         + focal_weight * torchvision.ops.focal_loss.sigmoid_focal_loss(
               logits, targets, alpha, gamma, reduction)

Verified params (docs/sydorskyi-config-verified.md § Loss):
    alpha=0.25, gamma=2.0, bce_weight=1.0, focal_weight=1.0, reduction="mean"

torchvision's `sigmoid_focal_loss` expects RAW LOGITS (it applies sigmoid
internally), matching our head's `clipwise_logits_long` output.

Confidence: VERIFIED (Phase A.2). 2026-05-28.
"""

import torch.nn as nn
import torchvision
from torch import Tensor


class FocalLossBCE(nn.Module):
    """1.0·BCEWithLogits + 1.0·sigmoid_focal(alpha, gamma).

    Args:
        alpha: focal loss alpha (Sydorskyi default 0.25).
        gamma: focal loss gamma (Sydorskyi default 2.0).
        reduction: "mean" (Sydorskyi default; use_weights=False) | "sum" | "none".
        bce_weight: weight on the plain BCE term (Sydorskyi default 1.0).
        focal_weight: weight on the focal term (Sydorskyi default 1.0).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """inputs = raw logits (B, num_class); targets = multi-hot float (B, num_class)."""
        focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(inputs, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss
