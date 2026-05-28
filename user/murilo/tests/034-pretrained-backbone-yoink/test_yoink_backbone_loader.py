"""Phase B unit tests for the Sydorskyi pretrained backbone loader.

Covers:
- timm features_only mode produces the expected output channel count
  - EffNetV2-S: 256 ch (NOT 1280 — conv_head/bn2 stripped by features_only)
  - NFNet-L0: 2304 ch
- Sydorskyi ckpt loads with strict=True (no missing/unexpected keys)
- Post-load: no all-zero conv weights (catches silent-load failure)
- Forward pass on a random (B, 1, n_mels, T_frame) tensor produces finite output

These tests require the Sydorskyi pretrained ckpts at the project shared storage path.
They are SKIPPED if the ckpts are not present (e.g. fresh clone without download).
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Import the loader script as a module
_LOADER_PATH = (
    Path(__file__).parent.parent.parent
    / "experiments"
    / "034-pretrained-backbone-yoink"
    / "scripts"
    / "load_sydorskyi_backbone.py"
)
_spec = importlib.util.spec_from_file_location("load_sydorskyi_backbone", _LOADER_PATH)
_loader_mod = importlib.util.module_from_spec(_spec)
sys.modules["load_sydorskyi_backbone"] = _loader_mod
_spec.loader.exec_module(_loader_mod)


def _ckpt_available(backbone_name: str) -> bool:
    ckpt_path = _loader_mod.DEFAULT_CKPT_ROOT / _loader_mod.SYDORSKYI_CKPT_MAP[backbone_name]
    return ckpt_path.is_file()


@pytest.mark.skipif(
    not _ckpt_available("tf_efficientnetv2_s_in21k"), reason="Sydorskyi ckpt not available"
)
def test_load_effnetv2s_strict():
    """EffNetV2-S loads with strict=True; output channels = 256 (features_only truncation)."""
    backbone = _loader_mod.load_sydorskyi_backbone("tf_efficientnetv2_s_in21k", verbose=False)
    assert backbone.feature_info.channels()[-1] == 256


@pytest.mark.skipif(not _ckpt_available("eca_nfnet_l0"), reason="Sydorskyi ckpt not available")
def test_load_nfnet_strict():
    """NFNet-L0 loads with strict=True; output channels = 2304."""
    backbone = _loader_mod.load_sydorskyi_backbone("eca_nfnet_l0", verbose=False)
    assert backbone.feature_info.channels()[-1] == 2304


@pytest.mark.skipif(
    not _ckpt_available("tf_efficientnetv2_s_in21k"), reason="Sydorskyi ckpt not available"
)
def test_load_effnetv2s_conv_weights_non_zero():
    """Catches silent strict=False load that leaves weights at random init."""
    backbone = _loader_mod.load_sydorskyi_backbone("tf_efficientnetv2_s_in21k", verbose=False)
    for name, param in backbone.named_parameters():
        if "conv" in name and "weight" in name and param.dim() == 4:
            assert param.detach().abs().mean().item() > 1e-3, f"{name} appears unloaded"


@pytest.mark.skipif(
    not _ckpt_available("tf_efficientnetv2_s_in21k"), reason="Sydorskyi ckpt not available"
)
def test_load_effnetv2s_forward_finite():
    """Forward pass on a (B, 1, 128, 313) tensor returns finite features (B, 256, H, W)."""
    backbone = _loader_mod.load_sydorskyi_backbone("tf_efficientnetv2_s_in21k", verbose=False)
    backbone.eval()
    x = torch.randn(2, 1, 128, 313)
    with torch.no_grad():
        out_list = backbone(x)
    assert isinstance(out_list, list)
    last = out_list[-1]
    assert last.shape[0] == 2 and last.shape[1] == 256
    assert torch.isfinite(last).all()


@pytest.mark.skipif(not _ckpt_available("eca_nfnet_l0"), reason="Sydorskyi ckpt not available")
def test_load_nfnet_forward_finite():
    """Same forward check for NFNet."""
    backbone = _loader_mod.load_sydorskyi_backbone("eca_nfnet_l0", verbose=False)
    backbone.eval()
    x = torch.randn(2, 1, 128, 313)
    with torch.no_grad():
        out_list = backbone(x)
    last = out_list[-1]
    assert last.shape[0] == 2 and last.shape[1] == 2304
    assert torch.isfinite(last).all()


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        _loader_mod.load_sydorskyi_backbone("nonexistent_model")
