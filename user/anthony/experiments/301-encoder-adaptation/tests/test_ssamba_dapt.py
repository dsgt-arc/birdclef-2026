"""Validation gate for SSAMBA native DAPT (`birdclef_301.ssamba_dapt`).

Lighter than the EAT gate (SSAMBA reuses exp 103's proven masked-recon logic), but
front-loads the two SSAMBA-specific risks: (1) the **pos_embed geometry** — the
hand-rolled masked forward adds `pos_embed[:, 1:]` to the patch grid, which
shape-mismatches if the checkpoint's native time-dim differs from the wrapper's;
(2) **grad through the Mamba CUDA kernel** must work (not just inference). Run on a
GPU node via `sbatch sbatch/validate-ssamba.sbatch`.
"""

from __future__ import annotations

import os

import pytest
import torch

ssamba_dapt = pytest.importorskip("birdclef_301.ssamba_dapt")
from birdclef_301.ssamba_dapt import (  # noqa: E402
    _SSAMBAHeads,
    _generate_span_mask,
    ssamba_masked_forward,
)
from birdclef_300.registry import build_torch_encoder  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.expanduser("~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth")
ENC_CFG = {"checkpoint": CKPT, "model_size": "small", "input_tdim": 512, "device": DEVICE}


def _wav(B: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, 16000 * 5, generator=g)


@pytest.fixture(scope="module")
def parts():
    wrapper = build_torch_encoder("ssamba_hf", ENC_CFG, DEVICE, CKPT)
    backbone = wrapper.model
    backbone.train()
    cfgm = backbone.config
    heads = _SSAMBAHeads(cfgm.embed_dim, cfgm.fpatch * cfgm.tpatch).to(DEVICE)
    unfold = torch.nn.Unfold(kernel_size=(cfgm.fpatch, cfgm.tpatch),
                             stride=(cfgm.fpatch, cfgm.tpatch)).to(DEVICE)
    return {"wrapper": wrapper, "backbone": backbone, "heads": heads, "unfold": unfold}


def _mel(wrapper, B=2):
    fbank = wrapper._waveform_to_fbank(_wav(B).to(DEVICE))   # (B, T, F)
    return fbank.unsqueeze(1).transpose(2, 3)                # (B, 1, F, T)


def test_pos_embed_geometry(parts):
    """[risk #1] pos_embed length must equal num_patches (+CLS) at the eval geometry."""
    backbone, wrapper = parts["backbone"], parts["wrapper"]
    mel = _mel(wrapper)
    num_patches = backbone.patch_embed(mel).shape[1]
    pos_len = backbone.pos_embed.shape[1]
    print(f"\nSSAMBA geometry: mel={tuple(mel.shape)} num_patches={num_patches} "
          f"pos_embed={tuple(backbone.pos_embed.shape)} "
          f"cls={'yes' if backbone.cls_token is not None else 'no'}")
    cls = 1 if backbone.cls_token is not None else 0
    assert pos_len == num_patches + cls, \
        f"pos_embed {pos_len} != num_patches {num_patches} + cls {cls} — geometry mismatch"


def test_frontend_parity(parts):
    """fbank targets the model's expected (tdim, fdim); mel matches forward()."""
    backbone, wrapper = parts["backbone"], parts["wrapper"]
    fbank = wrapper._waveform_to_fbank(_wav(2).to(DEVICE))
    assert fbank.shape[1:] == (backbone.config.img_size[1], backbone.config.img_size[0]) \
        or fbank.shape[1] == wrapper._model_tdim, fbank.shape


def test_masked_forward_finite(parts):
    out = ssamba_masked_forward(parts["backbone"], parts["heads"], parts["unfold"],
                                _mel(parts["wrapper"]), {"mask_ratio": 0.4, "mask_mode": "span"})
    assert torch.isfinite(out["loss_mpg"]) and torch.isfinite(out["loss_nce"])
    assert out["mask_count"] > 0


def test_grad_through_mamba(parts):
    """[risk #2] backward must flow through the Mamba kernel to the backbone."""
    backbone, heads = parts["backbone"], parts["heads"]
    backbone.zero_grad(); heads.zero_grad()
    out = ssamba_masked_forward(backbone, heads, parts["unfold"],
                                _mel(parts["wrapper"]), {"mask_ratio": 0.4, "mask_mode": "span"})
    (10.0 * out["loss_mpg"] + out["loss_nce"]).backward()
    gnorm = sum(p.grad.norm().item() for p in backbone.parameters() if p.grad is not None)
    assert gnorm > 0.0 and torch.isfinite(torch.tensor(gnorm)), gnorm


def test_state_dict_round_trip(parts):
    """backbone.state_dict() reloads into a fresh SSAMBAHFEncoder.model with no drift."""
    sd = parts["backbone"].state_dict()
    fresh = build_torch_encoder("ssamba_hf", ENC_CFG, DEVICE, CKPT)
    missing, unexpected = fresh.model.load_state_dict(sd, strict=False)
    assert not missing, f"missing: {missing[:5]}"
    assert not unexpected, f"unexpected: {unexpected[:5]}"


def test_tiny_overfit(parts):
    """Fixed batch + fixed mask → MPG MSE drops (can-it-learn, grad sanity)."""
    backbone, heads, unfold, wrapper = (parts["backbone"], parts["heads"],
                                        parts["unfold"], parts["wrapper"])
    mel = _mel(wrapper)
    num_patches = backbone.patch_embed(mel).shape[1]
    mask = _generate_span_mask(num_patches, 0.4, 5, 15, DEVICE)  # fixed
    params = list(backbone.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    losses = []
    for _ in range(40):
        out = ssamba_masked_forward(backbone, heads, unfold, mel,
                                    {"mask_ratio": 0.4}, mask_bool=mask)
        loss = 10.0 * out["loss_mpg"] + out["loss_nce"]
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(out["loss_mpg"].item())
    assert losses[-1] < 0.7 * losses[0], f"overfit failed: {losses[0]:.4f} -> {losses[-1]:.4f}"
