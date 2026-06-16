"""Exp 038 Stage 2 — warm-start transfer tests for train_finetune.transfer_pretrained_weights.

Two behaviours:
  - encoder_only=True (Stage-2 default): ONLY `encoder.*` (backbone) params copy from the pretrain
    checkpoint; the entire head + distill_head keep their fresh init (re-init the 234-class head,
    isolating the backbone). Critically, the class-count of source (2,683) and target (234) differ.
  - encoder_only=False: exp 032's species-aware transfer is preserved (full-copy shape matches +
    partial-copy the classifier rows for overlapping species). Regression guard for that path.

Uses a tiny stub LightningModule (no timm/backbone) so the transfer LOGIC is tested directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

_EXP_DIR = Path(__file__).parent.parent.parent / "experiments" / "038-xc-pretraining"
sys.path.insert(0, str(_EXP_DIR))

import train_finetune  # noqa: E402

transfer_pretrained_weights = train_finetune.transfer_pretrained_weights


class _Encoder(nn.Module):
    """Backbone stub: params are independent of num_classes (mimics EffNetV2-S keys)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, kernel_size=3)
        self.bn = nn.BatchNorm1d(4)


class _Head(nn.Module):
    """SED head stub: a num_classes-independent projection + a class-dim final layer."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(8, 6)  # shape-stable across class counts
        self.cls = nn.Linear(6, num_classes)  # cls.weight first-dim == num_classes


class _StubModule(pl.LightningModule):
    """Mirrors DistilledSEDFinetuneModule's submodule names: encoder.* / sed_head.* / distill_head.*."""

    def __init__(self, num_classes: int, with_distill: bool = True):
        super().__init__()
        self.encoder = _Encoder()
        self.sed_head = _Head(num_classes)
        if with_distill:
            self.distill_head = nn.Linear(
                4, 16
            )  # KD branch — fresh on the target, absent on source


def _save_ckpt(module: nn.Module, tmp_path: Path) -> str:
    p = tmp_path / "pretrain.ckpt"
    torch.save({"state_dict": module.state_dict()}, p)
    return str(p)


def test_encoder_only_copies_backbone_and_reinits_head(tmp_path):
    """encoder_only: encoder.* full-copied; head + distill_head untouched; strict load OK."""
    torch.manual_seed(0)
    src = _StubModule(num_classes=2683, with_distill=False)  # pretrain: KD off, 2683 classes
    torch.manual_seed(1)
    dst = _StubModule(num_classes=234, with_distill=True)  # finetune: KD on, 234 classes

    head_before = {k: v.clone() for k, v in dst.sed_head.state_dict().items()}
    distill_before = {k: v.clone() for k, v in dst.distill_head.state_dict().items()}

    transfer_pretrained_weights(dst, _save_ckpt(src, tmp_path), None, None, encoder_only=True)

    # Encoder fully transferred (every backbone param now equals the source).
    for k, v in src.encoder.state_dict().items():
        assert torch.equal(dst.encoder.state_dict()[k], v), f"encoder.{k} not copied"

    # Head re-init: every head param is unchanged from the target's fresh init (NOT copied), incl.
    # the shape-matching projection (proj) — encoder_only must not leak head weights.
    for k, v in dst.sed_head.state_dict().items():
        assert torch.equal(v, head_before[k]), f"sed_head.{k} should be re-init, got overwritten"

    # distill_head (KD branch, absent on the source) stays fresh.
    for k, v in dst.distill_head.state_dict().items():
        assert torch.equal(v, distill_before[k]), f"distill_head.{k} should be untouched"


def test_encoder_only_handles_class_count_mismatch(tmp_path):
    """The whole point: source (2683) and target (234) head shapes differ — must not raise."""
    src = _StubModule(num_classes=2683, with_distill=False)
    dst = _StubModule(num_classes=234, with_distill=True)
    # Should complete (internally load_state_dict(strict=True)) without error.
    transfer_pretrained_weights(dst, _save_ckpt(src, tmp_path), None, None, encoder_only=True)
    # cls layer keeps target width 234.
    assert dst.sed_head.cls.weight.shape[0] == 234


def test_species_aware_partial_copy_preserved(tmp_path):
    """Regression guard: encoder_only=False keeps exp 032's overlapping-species classifier transfer."""
    torch.manual_seed(0)
    src = _StubModule(num_classes=4, with_distill=False)
    torch.manual_seed(1)
    dst = _StubModule(num_classes=3, with_distill=True)

    src_species = ["a", "b", "c", "d"]
    dst_species = ["b", "c", "e"]  # overlap {b,c}; 'e' is target-only

    cls_e_before = dst.sed_head.cls.weight[2].clone()  # row for 'e' (target idx 2)

    transfer_pretrained_weights(
        dst, _save_ckpt(src, tmp_path), src_species, dst_species, encoder_only=False
    )

    # Overlapping species rows copied from the matching source index.
    assert torch.equal(dst.sed_head.cls.weight[0], src.sed_head.cls.weight[1])  # b: tgt0 <- src1
    assert torch.equal(dst.sed_head.cls.weight[1], src.sed_head.cls.weight[2])  # c: tgt1 <- src2
    # Target-only species row keeps its init.
    assert torch.equal(dst.sed_head.cls.weight[2], cls_e_before)
    # Encoder + shape-matching projection still full-copied in this mode.
    assert torch.equal(dst.encoder.conv.weight, src.encoder.conv.weight)
    assert torch.equal(dst.sed_head.proj.weight, src.sed_head.proj.weight)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
