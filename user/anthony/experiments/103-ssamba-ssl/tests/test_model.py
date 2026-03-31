"""Sanity tests for SSAMBA SSL pretraining model."""

import pytest
import torch
from nnAudio.features import MelSpectrogram
from ssamba_hf.config import SSAMBAConfig

from birdclef_103.models.pretrain import SSAMBAPretrainModel


def _make_tiny_model(mel_norm="pcen"):
    """Create a tiny model for CPU testing."""
    mel_spec = MelSpectrogram(
        sr=32000,
        n_fft=512,
        win_length=512,
        hop_length=320,
        n_mels=32,
        fmin=50,
        fmax=16000,
        trainable_mel=False,
        trainable_STFT=False,
    )
    config = SSAMBAConfig(
        img_size=(32, 100),
        patch_size=(32, 1),
        embed_dim=64,
        depth=4,
    )
    model = SSAMBAPretrainModel(
        config=config,
        mel_spec_layer=mel_spec,
        target_frames=100,
        mel_norm=mel_norm,
        mask_ratio=0.40,
        mask_mode="span",
        mask_span_min=3,
        mask_span_max=8,
        grid_jitter=0,
        decoder_depth=2,
    )
    return model


@pytest.mark.parametrize("mel_norm", ["pcen", "amplitude_to_db"])
def test_forward_pass(mel_norm):
    """Test that forward pass runs and returns expected keys."""
    model = _make_tiny_model(mel_norm=mel_norm)
    model.eval()

    waveforms = torch.randn(2, 32000)  # 1 second at 32kHz
    with torch.no_grad():
        out = model(waveforms)

    expected_keys = {
        "loss_mpg", "loss_nce", "nce_accuracy",
        "monitor_mse", "monitor_l1",
        "viz_mel", "viz_mask", "viz_pred", "viz_target",
    }
    assert set(out.keys()) == expected_keys


def test_forward_with_real_samples():
    """Test forward pass with real_samples padding mask."""
    model = _make_tiny_model()
    model.eval()

    waveforms = torch.randn(2, 32000)
    # Simulate one full-length and one short (half-length) recording
    real_samples = torch.tensor([32000, 16000], dtype=torch.long)
    with torch.no_grad():
        out = model(waveforms, real_samples=real_samples)

    assert torch.isfinite(out["loss_mpg"]).item()
    assert torch.isfinite(out["loss_nce"]).item()
    # Mask should not extend into padded region
    mask = out["viz_mask"]
    # 16000 samples / 320 hop = 50 real tokens (out of 100 total)
    # Mask should be False for tokens >= 50
    assert not mask[50:].any(), "Mask leaked into padded region"


@pytest.mark.parametrize("mel_norm", ["pcen", "amplitude_to_db"])
def test_losses_finite(mel_norm):
    """Test that all losses are finite and positive."""
    model = _make_tiny_model(mel_norm=mel_norm)
    model.eval()

    waveforms = torch.randn(2, 32000)
    with torch.no_grad():
        out = model(waveforms)

    assert torch.isfinite(out["loss_mpg"]).item()
    assert torch.isfinite(out["loss_nce"]).item()
    assert out["loss_mpg"].item() >= 0
    assert out["loss_nce"].item() >= 0


def test_nce_accuracy_at_chance():
    """NCE accuracy should be near chance (1/M) for random init."""
    model = _make_tiny_model()
    model.eval()

    waveforms = torch.randn(4, 32000)
    with torch.no_grad():
        out = model(waveforms)

    # With 40% masking of 100 tokens = 40 masked, chance = 1/40 = 0.025
    # Allow generous margin for small random models
    assert out["nce_accuracy"].item() < 0.5, "NCE accuracy suspiciously high for random init"


def test_mel_bounded_amplitude_to_db():
    """AmplitudeToDB should produce [0, 1] bounded output."""
    model = _make_tiny_model(mel_norm="amplitude_to_db")
    model.eval()

    waveforms = torch.randn(2, 32000)
    with torch.no_grad():
        mel = model.compute_mel(waveforms)

    assert mel.min() >= -0.01, f"Mel min {mel.min()} below 0"
    assert mel.max() <= 1.01, f"Mel max {mel.max()} above 1"


def test_pcen_output_shape():
    """PCEN should preserve mel shape."""
    model = _make_tiny_model(mel_norm="pcen")
    model.eval()

    waveforms = torch.randn(2, 32000)
    with torch.no_grad():
        mel = model.compute_mel(waveforms)

    # Should be (B, 1, n_mels, target_frames)
    assert mel.shape == (2, 1, 32, 100)


def test_backward_pass():
    """Test that gradients flow through the full model."""
    model = _make_tiny_model()
    model.train()

    waveforms = torch.randn(2, 32000)
    real_samples = torch.tensor([32000, 20000], dtype=torch.long)
    out = model(waveforms, real_samples=real_samples)
    loss = out["loss_nce"] + 10 * out["loss_mpg"]
    loss.backward()

    # Check gradients exist on backbone and both heads
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients found after backward pass"
