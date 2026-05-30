"""Soft-label pairwise AUC-surrogate loss (exp 037 — Noisy-Student soft PL).

A differentiable Wilcoxon–Mann–Whitney AUC surrogate generalized to SOFT targets.
For each class c, over all batch pairs (i, j) where sample i is "more positive"
than j (y_i > y_j), penalize ranking violations (s_i < s_j) with a logistic
margin, pair-weighted by (y_i - y_j). Reduces to the standard pairwise AUC
surrogate when targets are binary.

    D[i,j,c] = s[i,c] - s[j,c]                    # score diff (want > 0)
    W[i,j,c] = relu(y[i,c] - y[j,c])              # soft pair weight
    loss_c   = sum_ij W * softplus(-D) / sum_ij W # per-class, then macro-mean

Motivation (deep research, 2025 winner): the competition metric is macro
ROC-AUC; SoftAUC optimizes ranking directly and tolerates soft pseudo-labels,
whereas BCE optimizes calibration. O(B^2 * C) per call (~1M ops at B=64,
C=234) — computed in float32 for stability under AMP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_auc_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Macro soft-AUC surrogate.

    Args:
        logits: (B, C) raw scores (higher = more positive).
        targets: (B, C) soft labels in [0, 1].
        eps: classes whose in-batch pair-weight mass <= eps are skipped (no
            ranking signal — e.g. all samples share the same label for that class).

    Returns:
        Scalar loss. 0 (graph-connected) if no class has label variation in the batch.
    """
    s = logits.float()
    y = targets.float()
    # Pairwise over the batch dim → (B, B, C). D[i,j,c]=s[i,c]-s[j,c]; W[i,j,c]=relu(y[i,c]-y[j,c]).
    D = s.unsqueeze(1) - s.unsqueeze(0)
    W = (y.unsqueeze(1) - y.unsqueeze(0)).clamp(min=0.0)
    pen = F.softplus(-D)  # large when s_i << s_j (violation), ~0 when s_i >> s_j
    num = (W * pen).sum(dim=(0, 1))  # (C,)
    den = W.sum(dim=(0, 1))  # (C,)
    valid = den > eps
    if not torch.any(valid):
        return s.sum() * 0.0  # keep autograd graph; no ranking signal this batch
    return (num[valid] / den[valid]).mean()


class SoftAUCLoss(nn.Module):
    """nn.Module wrapper around `soft_auc_loss` (so it composes like FocalLossBCE)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_auc_loss(logits, targets, self.eps)
