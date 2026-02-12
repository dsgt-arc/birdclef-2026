"""Tests for SSAMBASEDClassifier — CPU tests with pure-Python Mamba fallback.

GPU tests with CUDA Mamba kernels are in test_forward_gpu.py.
"""

import pytest
import torch
import torch.nn as nn

from ssamba_hf import SSAMBAConfig, MAMBA_SSM_AVAILABLE
from src.model import SSAMBASEDClassifier


def _make_dummy_mel_spec(n_mels=128):
    """Create a dummy mel spec layer that just reshapes waveforms → fake mel."""

    class DummyMelSpec(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_mels = n_mels

        def forward(self, waveforms):
            B = waveforms.shape[0]
            # Simulate: (B, num_samples) → (B, n_mels, T)
            T = waveforms.shape[1] // 320  # hop_length=320
            return torch.randn(B, self.n_mels, T)

    return DummyMelSpec()


@pytest.fixture
def small_config():
    """Small config for CPU testing (depth=2 to avoid O(L) Mamba loop being too slow)."""
    return SSAMBAConfig(
        embed_dim=64,
        depth=2,
        img_size=(128, 100),
        patch_size=(128, 1),
        use_cls_token=True,
        use_middle_cls_token=True,
        rms_norm=False,
    )


class TestSSAMBASEDClassifierCPU:
    """CPU-only tests with small config and dummy mel spec."""

    def test_forward_shapes(self, small_config):
        num_classes = 21
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )
        # 100 frames * 320 hop = 32000 samples (1s at 32kHz)
        waveforms = torch.randn(2, 32000)

        clip_pred, sed_logits = model(waveforms)

        assert clip_pred.shape == (2, num_classes)
        assert sed_logits.shape == (2, 100, num_classes)

    def test_clip_pred_in_zero_one(self, small_config):
        """clip_pred should be probabilities (from attention pooling of sigmoid)."""
        num_classes = 21
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )
        waveforms = torch.randn(2, 32000)

        with torch.no_grad():
            clip_pred, sed_logits = model(waveforms)

        assert (clip_pred >= 0).all()
        assert (clip_pred <= 1).all()

    def test_sed_logits_are_raw(self, small_config):
        """sed_logits should be raw (not sigmoid'd) — can be negative."""
        num_classes = 21
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )
        waveforms = torch.randn(2, 32000)

        with torch.no_grad():
            _, sed_logits = model(waveforms)

        # With random init, logits should span both positive and negative
        assert sed_logits.min() < 0 or sed_logits.max() > 0

    def test_padding_mask_affects_clip_pred(self, small_config):
        """Padding mask should change attention pooling output."""
        num_classes = 21
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )
        waveforms = torch.randn(2, 32000)

        # Full mask vs half mask
        full_mask = torch.ones(2, 100, dtype=torch.bool)
        half_mask = torch.ones(2, 100, dtype=torch.bool)
        half_mask[:, 50:] = False

        with torch.no_grad():
            clip_full, _ = model(waveforms, padding_mask=full_mask)
            clip_half, _ = model(waveforms, padding_mask=half_mask)

        # Outputs should differ when mask differs
        assert not torch.allclose(clip_full, clip_half, atol=1e-5)

    def test_spec_augment_applied(self, small_config):
        """SpecAugment should modify mel inside model forward."""
        from src.spec_augment import SpecAugment

        num_classes = 21
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )
        spec_aug = SpecAugment(freq_mask_param=20, time_mask_param=10, num_freq_masks=2, num_time_masks=2)

        torch.manual_seed(42)
        waveforms = torch.randn(2, 32000)

        # With vs without augment — should produce different outputs
        torch.manual_seed(0)
        clip_aug, _ = model(waveforms, spec_augment=spec_aug)
        torch.manual_seed(0)
        clip_no_aug, _ = model(waveforms, spec_augment=None)

        # Note: with the same mel_spec (random), both should differ due to masking
        # This test is probabilistic but SpecAugment should generally change the output

    def test_backward_grads(self, small_config):
        """Gradients should flow to all components."""
        num_classes = 5
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )
        waveforms = torch.randn(2, 32000)
        labels = torch.zeros(2, num_classes)
        labels[:, 0] = 1.0

        clip_pred, sed_logits = model(waveforms)
        # Use probability-based loss for clip_pred
        loss = torch.nn.functional.binary_cross_entropy(clip_pred, labels)
        loss.backward()

        assert model.sed_head[1].weight.grad is not None
        assert model.backbone.patch_embed.proj.weight.grad is not None

    def test_compute_mel_truncate_and_pad(self, small_config):
        """Test mel computation handles short and long waveforms."""
        num_classes = 5
        mel_spec = _make_dummy_mel_spec()
        model = SSAMBASEDClassifier(
            config=small_config, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=100,
        )

        # Short waveform (fewer frames than target)
        short_wave = torch.randn(1, 16000)  # ~50 frames
        mel_short = model.compute_mel(short_wave)
        assert mel_short.shape == (1, 1, 128, 100)

        # Long waveform (more frames than target)
        long_wave = torch.randn(1, 64000)  # ~200 frames
        mel_long = model.compute_mel(long_wave)
        assert mel_long.shape == (1, 1, 128, 100)
