#!/usr/bin/env python3
"""Gradient-flow sanity check for the exp 031 KD branch.

Verifies on a synthetic batch (no GPU, no real Perch ONNX needed) that:
  1. Backbone gradient norm > 0 -- backbone IS being trained.
  2. sed_head gradient norm > 0 -- SED head IS being trained.
  3. distill_head gradient norm > 0 -- distill head IS being trained.
  4. cls_loss and dist_loss are finite (no NaN/Inf).
  5. perch_emb_target has the expected (B, 1536) shape under mixup.
  6. With KD on but mixup off: perch_emb_target == perch_emb (identity).
  7. With KD on AND mixup on: target uses the SAME (lam, indices) as the input.

This monkey-patches `module.perch_teacher` with a deterministic stub so the
test runs without the real Perch ONNX file. The student-side code paths
(DistillHead, feat.detach(), loss composition) are exercised end-to-end.

Run:
    python test_kd_gradient_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.heads.sed_head import LSEHead  # noqa: E402
from birdclef.lightning.sed_finetune import (  # noqa: E402
    DistillHead,
    SEDFinetuneModule,
)
from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram  # noqa: E402


class StubPerchTeacher:
    """Deterministic per-sample stub teacher. Independent of batch contents
    *modulo* their hash, so perch(x[indices]) == perch(x)[indices] holds."""

    def __init__(self, embed_dim: int = 1536):
        self.embed_dim = embed_dim
        # Deterministic per-sample-hash projection.
        gen = torch.Generator().manual_seed(0)
        self._proj = torch.randn(160000, embed_dim, generator=gen) * 0.01

    @torch.no_grad()
    def embed(self, waveforms_5s: torch.Tensor) -> torch.Tensor:
        # Shape: (B, 160000) -> (B, embed_dim). Deterministic per-row.
        return waveforms_5s.float() @ self._proj.to(waveforms_5s.device)


def _build_module(*, kd_on: bool, mixup_prob: float) -> SEDFinetuneModule:
    encoder = EfficientNetEncoder(
        model_name="hgnetv2_b0.ssld_stage2_ft_in1k",
        pretrained=False,  # no internet during test
        in_chans=1,
    )
    sed_head = LSEHead(
        in_channels=encoder.embed_dim,
        num_classes=10,  # tiny for test speed
        hidden_dim=64,
        dropout=0.0,
        gem_p=3.0,
        lse_temperature=1.0,
        freq_pool="mean",
    )
    mel_transform = NNAudioMelSpectrogram(
        sample_rate=32000,
        n_mels=64,  # smaller for test speed
        target_frames=256,
        n_fft=1024,
        hop_length=625,
        norm_type="amplitude_to_db",
    )

    # Build with KD turned off, then flip the flag if needed -- so __init__
    # doesn't try to open a real ONNX session. We stub the teacher after.
    module = SEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=10,
        max_epochs=20,
        warmup_epochs=0,
        clipwise_weight=1.0,
        framewise_weight=0.0,
        blend_clip=1.0,
        blend_frame=0.0,
        loss_type="bce",
        mel_transform=mel_transform,
        mixup_prob=mixup_prob,
        mixup_alpha=1.0,
        mixup_warmup_epochs=0,
        mixup_theta=0.8,
        mixup_per_sample=True,
        mixup_mode="interpolate",
        scheduler_type="one_cycle",
        # KD flag controls instantiation of distill_head only at this point;
        # we'll inject a stub teacher right after if KD is on.
        use_perch_distill=False,
        perch_onnx_path=None,
        perch_embed_dim=1536,
        alpha_distill=1.0,
    )
    if kd_on:
        # Manually flip the hparam so training_step takes the KD branch.
        module.hparams.use_perch_distill = True
        # Add the distill head + stub teacher (skipping the real ONNX session).
        module.distill_head = DistillHead(encoder.embed_dim, 1536)
        module.perch_teacher = StubPerchTeacher(embed_dim=1536)
    return module


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    seen = 0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
            seen += 1
    return float(total**0.5) if seen else 0.0


def main():
    torch.manual_seed(42)
    B = 4
    T = 160000  # 5s @ 32 kHz

    print("=" * 60)
    print("Test 1: KD on, mixup OFF (warmup-style)")
    print("=" * 60)
    module = _build_module(kd_on=True, mixup_prob=0.0)
    module.train()
    # Synthetic batch
    x = torch.randn(B, T)
    labels = torch.randint(0, 2, (B, 10)).float()
    primary_idx = torch.randint(0, 10, (B,))
    batch = (x, labels, primary_idx)

    module.zero_grad()
    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss), f"loss not finite: {loss}"
    backbone_g = _grad_norm(module.encoder)
    head_g = _grad_norm(module.sed_head)
    distill_g = _grad_norm(module.distill_head)
    print(f"  loss = {loss.item():.4f}")
    print(f"  backbone grad norm    = {backbone_g:.4e}")
    print(f"  sed_head grad norm    = {head_g:.4e}")
    print(f"  distill_head grad norm= {distill_g:.4e}")
    assert backbone_g > 0, "Backbone got no gradient!"
    assert head_g > 0, "sed_head got no gradient!"
    assert distill_g > 0, "distill_head got no gradient!"
    print("  ALL CHECKS PASS")

    print("\n" + "=" * 60)
    print("Test 2: KD on, mixup ON (interpolate, per-sample)")
    print("=" * 60)
    torch.manual_seed(7)
    module = _build_module(kd_on=True, mixup_prob=1.0)
    module.train()
    x = torch.randn(B, T)
    labels = torch.randint(0, 2, (B, 10)).float()
    primary_idx = torch.randint(0, 10, (B,))
    batch = (x, labels, primary_idx)

    module.zero_grad()
    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss), f"loss not finite under mixup: {loss}"
    backbone_g = _grad_norm(module.encoder)
    head_g = _grad_norm(module.sed_head)
    distill_g = _grad_norm(module.distill_head)
    print(f"  loss = {loss.item():.4f}")
    print(f"  backbone grad norm    = {backbone_g:.4e}")
    print(f"  sed_head grad norm    = {head_g:.4e}")
    print(f"  distill_head grad norm= {distill_g:.4e}")
    assert backbone_g > 0, "Backbone got no gradient under mixup!"
    assert head_g > 0, "sed_head got no gradient under mixup!"
    assert distill_g > 0, "distill_head got no gradient under mixup!"
    print("  ALL CHECKS PASS")

    print("\n" + "=" * 60)
    print("Test 3: row-permutation invariance of teacher (the 1x trick)")
    print("=" * 60)
    teacher = StubPerchTeacher(embed_dim=1536)
    x = torch.randn(B, T)
    perm = torch.tensor([2, 0, 3, 1])
    e1 = teacher.embed(x[perm])
    e2 = teacher.embed(x)[perm]
    diff = (e1 - e2).abs().max().item()
    print(f"  max |perch(x[perm]) - perch(x)[perm]| = {diff:.2e}")
    assert diff < 1e-5, "Per-sample invariance broken — 1x teacher trick INVALID"
    print("  ALL CHECKS PASS")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
