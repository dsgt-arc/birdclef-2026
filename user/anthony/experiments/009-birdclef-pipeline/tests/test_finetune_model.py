"""Tests for clip-level fine-tuning model."""

import torch

from birdclef_009.models.finetune import SSAMBAClassifier


def test_forward_output_shape(tiny_config, mel_spec_layer):
    """Classifier forward pass produces correct output shape."""
    num_classes = 10
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
    )
    model.eval()

    B = 2
    waveforms = torch.randn(B, 320000)

    with torch.no_grad():
        clip_logits = model(waveforms)

    assert clip_logits.shape == (B, num_classes), f"Expected ({B}, {num_classes}), got {clip_logits.shape}"


def test_clip_pred_finite(tiny_config, mel_spec_layer):
    """Clip logits should be finite."""
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=5,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
    )
    model.eval()

    waveforms = torch.randn(2, 320000)

    with torch.no_grad():
        clip_logits = model(waveforms)

    assert torch.isfinite(clip_logits).all()


def test_with_padding_mask(tiny_config, mel_spec_layer):
    """Model should handle padding masks correctly."""
    num_classes = 5
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
    )
    model.eval()

    B = 2
    waveforms = torch.randn(B, 320000)
    padding_mask = torch.ones(B, 1000, dtype=torch.bool)
    padding_mask[0, 800:] = False

    with torch.no_grad():
        clip_logits = model(waveforms, padding_mask=padding_mask)

    assert clip_logits.shape == (B, num_classes)
    assert torch.isfinite(clip_logits).all()


def test_forward_windows(tiny_config, mel_spec_layer):
    """forward_windows produces two 5s windows per 10s input."""
    num_classes = 5
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
    )
    model.eval()

    waveforms = torch.randn(1, 320000)
    window_logits = model.forward_windows(waveforms, window_size=500)

    assert window_logits.shape == (1, 2, num_classes)
    assert torch.isfinite(window_logits).all()
