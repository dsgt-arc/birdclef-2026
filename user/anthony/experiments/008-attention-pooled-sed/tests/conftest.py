"""Shared fixtures for experiment 008 tests."""

import os
from pathlib import Path

import pytest
import torch
from ssamba_hf import SSAMBAConfig, MAMBA_SSM_AVAILABLE

# ── Skip markers ──────────────────────────────────────────────────────────

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
requires_mamba_ssm = pytest.mark.skipif(
    not MAMBA_SSM_AVAILABLE,
    reason="mamba-ssm not installed",
)

PRETRAINED_CKPT = Path(
    os.environ.get(
        "SSAMBA_CKPT",
        os.path.expanduser("~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_tiny_400.pth"),
    )
)
requires_pretrained = pytest.mark.skipif(
    not PRETRAINED_CKPT.exists(),
    reason=f"Pre-trained checkpoint not found at {PRETRAINED_CKPT}",
)


# ── Configs ───────────────────────────────────────────────────────────────

@pytest.fixture
def config_128x1():
    """SSAMBA Tiny config with 128x1 patches for 10s audio."""
    return SSAMBAConfig(
        embed_dim=192,
        depth=24,
        img_size=(128, 1000),
        patch_size=(128, 1),
        use_cls_token=True,
        use_middle_cls_token=True,
        rms_norm=False,
    )


@pytest.fixture
def pretrained_ckpt_path():
    """Path to pre-trained SSAMBA Tiny checkpoint."""
    return PRETRAINED_CKPT


@pytest.fixture
def num_classes():
    """HSN has 21 classes."""
    return 21
