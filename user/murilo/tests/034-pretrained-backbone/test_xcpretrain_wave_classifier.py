"""Phase B unit tests for XCPretrainWaveClassifier — end-to-end orchestrator.

Covers:
- Construction from default config (EffNetV2-S, 234 classes)
- extract_spec produces the expected (B, 1, n_mels, T_frame) shape for 5s @ 32kHz input
- Full forward returns the expected output dict with finite values, correct shapes
- spec_augment_config raises NotImplementedError (deferred to next Phase B sub-commit)
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier

# Import the loader script as a module (shared with test_xcpretrain_backbone_loader.py)
_LOADER_PATH = (
    Path(__file__).parent.parent.parent
    / "experiments"
    / "034-pretrained-backbone"
    / "scripts"
    / "load_sydorskyi_backbone.py"
)
_spec = importlib.util.spec_from_file_location("load_sydorskyi_backbone", _LOADER_PATH)
_loader_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("load_sydorskyi_backbone", _loader_mod)
_spec.loader.exec_module(_loader_mod)


def _ckpt_available(backbone_name: str) -> bool:
    ckpt_path = _loader_mod.DEFAULT_CKPT_ROOT / _loader_mod.SYDORSKYI_CKPT_MAP[backbone_name]
    return ckpt_path.is_file()


def test_construct_default():
    model = XCPretrainWaveClassifier(
        backbone="tf_efficientnetv2_s_in21k",
    )
    # Mel chain has 3 components
    assert len(model.logmelspec_extractor) == 3
    # Backbone is in features_only mode (output channels = 256 for EffNetV2-S)
    assert model.backbone.feature_info.channels()[-1] == 256
    # Head exists, num_class default = 234
    assert model.head.attention.out_channels == 234


def test_extract_spec_shape_5s_32khz():
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        spec = model.extract_spec(wav)
    assert spec.shape == (2, 1, 128, 313), f"got {spec.shape}"
    assert torch.isfinite(spec).all()
    # After NormalizeMelSpec, per-sample range = [0, 1]
    assert spec.min().item() == pytest.approx(0.0, abs=1e-4)
    assert spec.max().item() == pytest.approx(1.0, abs=1e-4)


def test_forward_output_keys_and_shapes():
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        out = model(wav)
    assert set(out.keys()) == {
        "clipwise_pred_long",
        "clipwise_logits_long",
        "framewise_logits_long",
    }
    assert out["clipwise_pred_long"].shape == (2, 234)
    assert out["clipwise_logits_long"].shape == (2, 234)
    # framewise time dim depends on backbone (10 for EffNetV2-S features_only at 5s/32k/hop=512)
    assert (
        out["framewise_logits_long"].shape[0] == 2 and out["framewise_logits_long"].shape[2] == 234
    )


def test_forward_outputs_finite():
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        out = model(wav)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} non-finite"


def test_forward_with_precomputed_spec():
    """Bypass mel extraction if `spec` is provided."""
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    model.eval()
    spec = torch.rand(2, 1, 128, 313)
    with torch.no_grad():
        out = model(x=None, spec=spec)
    assert out["clipwise_logits_long"].shape == (2, 234)


def test_return_spec_feature_short_circuit():
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        spec = model(wav, return_spec_feature=True)
    assert spec.shape == (2, 1, 128, 313)


def test_return_cnn_emb_short_circuit():
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        emb = model(wav, return_cnn_emb=True)
    # EffNetV2-S features_only last stage: (B, 256, H_freq, W_time)
    assert emb.shape[0] == 2 and emb.shape[1] == 256


def test_spec_augment_config_raises_not_implemented():
    """spec_augment_config support is deferred — should raise clearly."""
    with pytest.raises(NotImplementedError, match="spec_augment_config support pending"):
        XCPretrainWaveClassifier(
            backbone="tf_efficientnetv2_s_in21k",
            spec_augment_config={"freq_mask": {"mask_max_length": 10}},
        )


def test_custom_num_class():
    model = XCPretrainWaveClassifier(
        backbone="tf_efficientnetv2_s_in21k",
        head_config={"p": 0.5, "num_class": 100, "train_period": 5.0, "infer_period": 5.0},
    )
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        out = model(wav)
    assert out["clipwise_logits_long"].shape == (2, 100)


@pytest.mark.skipif(
    not _ckpt_available("tf_efficientnetv2_s_in21k"),
    reason="Sydorskyi pretrained ckpt not available",
)
def test_load_pretrained_backbone_then_forward():
    """End-to-end: instantiate → load Sydorskyi ckpt → forward on random waveform."""
    model = XCPretrainWaveClassifier(backbone="tf_efficientnetv2_s_in21k")
    pretrained = _loader_mod.load_sydorskyi_backbone("tf_efficientnetv2_s_in21k", verbose=False)
    model.backbone.load_state_dict(pretrained.state_dict(), strict=True)
    model.eval()
    wav = torch.randn(2, 5 * 32000)
    with torch.no_grad():
        out = model(wav)
    assert out["clipwise_logits_long"].shape == (2, 234)
    assert torch.isfinite(out["clipwise_logits_long"]).all()
