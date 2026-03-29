"""Loss functions and data augmentation for ProtoSSM training.

Cross-ref: ref_yuriy_protossm_v5.py lines 531-548 (focal), 1660-1680 (mixup).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for multi-label classification.

    Reduces contribution of easy examples, focuses on hard ones.
    When gamma=0, equivalent to standard BCE.
    """
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    return loss


def mixup_files(
    emb: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    hours: np.ndarray,
    families: np.ndarray | None,
    alpha: float = 0.3,
) -> tuple:
    """File-level mixup augmentation for ProtoSSM training.

    Mixes pairs of files with random lambda from Beta(alpha, alpha).
    Lambda is enforced >= 0.5 so the dominant sample stays dominant.
    Discrete features (site, hour) keep the dominant sample's values.
    """
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours, families

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5

    perm = np.random.permutation(n)

    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]

    # For discrete features (site, hour), keep the dominant sample's values
    families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None

    return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix
