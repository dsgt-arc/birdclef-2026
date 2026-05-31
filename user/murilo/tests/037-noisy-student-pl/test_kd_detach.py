"""Exp 037 — kd_detach_head gradient-routing test (the cell-A vs cell-C lever).

The whole point of cell C is that PL labels reach the BACKBONE, not just the head.
This verifies the flag actually changes gradient flow: with alpha_distill=0 (so only
the classification loss can reach the backbone), `kd_detach_head=True` (cell A) leaves
the encoder with ~zero gradient (cls is detached), while `kd_detach_head=False` (cell C)
gives the encoder real gradient from the cls loss.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder
from birdclef.heads.sed_head import AttHeadLogitSpace
from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule

NUM_CLASSES = 234
BATCH = 2
N_MELS, TIME_FRAMES, SAMPLES_5S = 256, 313, 160_000


class _IdentityMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_mels, self.time_frames = N_MELS, TIME_FRAMES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        flat = x.float().reshape(B, -1)
        proj = flat[:, : self.n_mels * self.time_frames]
        if proj.shape[1] < self.n_mels * self.time_frames:
            proj = F.pad(proj, (0, self.n_mels * self.time_frames - proj.shape[1]))
        return proj.reshape(B, 1, self.n_mels, self.time_frames)


class _StubPerchTeacher:
    def __init__(self, *a, **k):
        pass

    @torch.no_grad()
    def embed(self, w):
        return torch.zeros(w.shape[0], 1536, device=w.device)


def _kd_module(kd_detach_head: bool, loss_type: str = "bce"):
    enc = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=enc.embed_dim, num_classes=NUM_CLASSES, hidden_dim=512)
    with patch("birdclef.lightning.distilled_sed_finetune.PerchTeacher", _StubPerchTeacher):
        return DistilledSEDFinetuneModule(
            encoder=enc,
            sed_head=head,
            num_classes=NUM_CLASSES,
            mel_transform=_IdentityMel(),
            use_perch_distill=True,
            perch_onnx_path="/stub.onnx",
            perch_embed_dim=1536,
            alpha_distill=0.0,  # isolate: only the cls loss can reach the backbone
            kd_detach_head=kd_detach_head,
            loss_type=loss_type,
        )


def _encoder_grad_norm(m) -> float:
    sq = sum(p.grad.norm().item() ** 2 for p in m.encoder.parameters() if p.grad is not None)
    return sq**0.5


def _batch():
    torch.manual_seed(0)
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, [0, 17]] = 1.0
    return (x, labels)


def test_detach_true_blocks_cls_grad_to_backbone():
    m = _kd_module(kd_detach_head=True)
    with patch.object(m, "log"):
        loss = m.training_step(_batch(), 0)
    loss.backward()
    assert _encoder_grad_norm(m) < 1e-5  # cls detached + distill weighted 0 → no backbone grad


def test_detach_false_lets_cls_train_backbone():
    m = _kd_module(kd_detach_head=False)
    with patch.object(m, "log"):
        loss = m.training_step(_batch(), 0)
    loss.backward()
    assert _encoder_grad_norm(m) > 1e-3  # cls loss flows into the backbone


def test_hparams_store_levers():
    m = _kd_module(kd_detach_head=False, loss_type="softauc")
    assert m.hparams.kd_detach_head is False
    assert m.hparams.loss_type == "softauc"


def test_softauc_loss_type_changes_cls_loss():
    """loss_type=softauc routes _dual_supervision_loss through the ranking loss."""
    m_bce = _kd_module(kd_detach_head=True, loss_type="bce")
    m_auc = _kd_module(kd_detach_head=True, loss_type="softauc")
    torch.manual_seed(1)
    clip = torch.randn(BATCH, NUM_CLASSES)
    frame = torch.randn(BATCH, NUM_CLASSES, 10)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, [0, 17]] = 1.0
    l_bce, _ = m_bce._dual_supervision_loss(clip, frame, labels)
    l_auc, _ = m_auc._dual_supervision_loss(clip, frame, labels)
    assert torch.isfinite(l_bce) and torch.isfinite(l_auc)
    assert not torch.isclose(l_bce, l_auc)  # different objectives
