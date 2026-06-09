"""Losses for encoder adaptation.

Multi-label focal BCE for supervised-focal LoRA (the shared finetune_lora loop is
single-label CrossEntropy), and a latent-bootstrap cosine loss for DAPT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    """Multi-label focal loss over independent per-species sigmoids.

    gamma=0 reduces to plain BCE-with-logits. Birds dominate the focal label
    space, so a small gamma down-weights the easy negatives.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.gamma > 0:
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            bce = bce * (1 - p_t).pow(self.gamma)
        return bce.mean()


def latent_bootstrap_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity between student (masked view) and teacher (full view).

    The data2vec / BYOL-style continued-pretraining objective used by DAPT: a
    masked student is trained to predict the full-clip teacher embedding, with the
    teacher tracked as an EMA of the student (no labels, target-domain audio only).
    """
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher.detach(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()
