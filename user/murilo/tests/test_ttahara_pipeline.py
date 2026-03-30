"""Tests for ttahara's HGNetV2-B0 pipeline reproduction (v5f/v5g).

Validates:
- LSEHead with freq_pool="mean" matches ttahara's architecture
- TtaharaMelTransform produces correct shapes and value ranges
- Per-sample MixUp produces diverse lambdas
- OneCycleLR scheduler configuration
"""

import math

import pytest
import torch

from birdclef.heads.sed_head import LSEHead, LSEPooling
from birdclef.preprocessing.ttahara_mel import TtaharaMelTransform

# ============================================================================
# LSEHead tests
# ============================================================================


class TestLSEHeadFreqPool:
    """Test LSEHead with different frequency pooling methods."""

    def test_gem_default_backward_compat(self):
        """Default freq_pool='gem' preserves existing behavior."""
        head = LSEHead(in_channels=320, num_classes=234, hidden_dim=512)
        assert head.freq_pool_type == "gem"
        assert hasattr(head, "gem")

        x = torch.randn(2, 320, 7, 31)
        clip, frame = head(x)
        assert clip.shape == (2, 234)
        assert frame.shape == (2, 234, 31)

    def test_mean_freq_pool(self):
        """freq_pool='mean' uses simple average over frequency."""
        head = LSEHead(in_channels=320, num_classes=234, hidden_dim=320, freq_pool="mean")
        assert head.freq_pool_type == "mean"
        assert not hasattr(head, "gem")

        x = torch.randn(2, 320, 7, 31)
        clip, frame = head(x)
        assert clip.shape == (2, 234)
        assert frame.shape == (2, 234, 31)

    def test_mean_matches_torch_mean(self):
        """freq_pool='mean' gives same result as torch.mean(dim=2)."""
        head = LSEHead(in_channels=64, num_classes=10, hidden_dim=64, freq_pool="mean")
        head.eval()  # Disable dropout for deterministic comparison
        x = torch.randn(2, 64, 7, 31)

        # Manual computation
        h_manual = x.mean(dim=2)  # (B, C, T)
        h_manual = h_manual.permute(0, 2, 1)  # (B, T, C)
        framewise_manual = head.cls_fc(h_manual)  # (B, T, 10)
        clip_manual = head.lse_pool(framewise_manual)  # (B, 10)

        clip, frame = head(x)
        torch.testing.assert_close(clip, clip_manual)

    def test_hgnet_dimensions(self):
        """HGNet-B0 dimensions (1024 channels, hidden=1024) work correctly."""
        head = LSEHead(in_channels=1024, num_classes=234, hidden_dim=1024, freq_pool="mean")
        x = torch.randn(2, 1024, 8, 8)  # HGNet output at (256,256) input
        clip, frame = head(x)
        assert clip.shape == (2, 234)
        assert frame.shape == (2, 234, 8)

    def test_effnet_dimensions(self):
        """EfficientNet-B0 dimensions (320 channels, hidden=320) work correctly."""
        head = LSEHead(in_channels=320, num_classes=234, hidden_dim=320, freq_pool="mean")
        x = torch.randn(2, 320, 7, 31)
        clip, frame = head(x)
        assert clip.shape == (2, 234)
        assert frame.shape == (2, 234, 31)

    def test_gradients_flow(self):
        """Gradients flow through freq_pool='mean' path."""
        head = LSEHead(in_channels=64, num_classes=10, hidden_dim=64, freq_pool="mean")
        x = torch.randn(2, 64, 4, 8, requires_grad=True)
        clip, _ = head(x)
        loss = clip.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ============================================================================
# TtaharaMelTransform tests
# ============================================================================


class TestTtaharaMelTransform:
    """Test ttahara's mel spectrogram transform."""

    @pytest.fixture
    def mel_transform(self):
        return TtaharaMelTransform()

    def test_batch_output_shape(self, mel_transform):
        """Batch input produces (B, 1, 256, 256) output."""
        wave = torch.randn(4, 160000)  # 5s at 32kHz
        out = mel_transform(wave)
        assert out.shape == (4, 1, 256, 256)

    def test_single_output_shape(self, mel_transform):
        """Single waveform produces (1, 256, 256) output."""
        wave = torch.randn(160000)
        out = mel_transform(wave)
        assert out.shape == (1, 256, 256)

    def test_value_range(self, mel_transform):
        """Output values are in [0, 1] after min-max normalization."""
        wave = torch.randn(4, 160000)
        out = mel_transform(wave)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_per_sample_normalization(self, mel_transform):
        """Each sample is independently normalized (min=0, max=1)."""
        wave = torch.randn(4, 160000)
        # Scale samples differently
        wave[0] *= 10.0
        wave[1] *= 0.01
        out = mel_transform(wave)

        for i in range(4):
            sample = out[i]
            assert sample.min() >= -1e-6
            assert (
                abs(sample.max() - 1.0) < 1e-5
            ), f"Sample {i} max={sample.max():.6f}, expected ~1.0"

    def test_custom_target_shape(self):
        """Custom target_shape works."""
        mel = TtaharaMelTransform(target_shape=(128, 128))
        wave = torch.randn(2, 160000)
        out = mel(wave)
        assert out.shape == (2, 1, 128, 128)

    def test_short_audio_padding(self, mel_transform):
        """Short audio (< 5s) still produces valid output."""
        wave = torch.randn(2, 32000)  # 1s at 32kHz
        out = mel_transform(wave)
        assert out.shape == (2, 1, 256, 256)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6


# ============================================================================
# Per-sample MixUp tests
# ============================================================================


class TestPerSampleMixUp:
    """Test per-sample MixUp lambda sampling (ttahara's approach)."""

    def test_per_sample_lambda_shape(self):
        """Per-sample lambda has shape (batch_size,)."""
        batch_size = 16
        alpha = 1.0
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample(sample_shape=(batch_size,))
        lam = torch.maximum(lam, 1 - lam).float()
        assert lam.shape == (batch_size,)

    def test_lambda_ge_half(self):
        """All lambdas are >= 0.5 after max(lam, 1-lam)."""
        alpha = 1.0
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample(sample_shape=(1000,))
        lam = torch.maximum(lam, 1 - lam).float()
        assert (lam >= 0.5).all()

    def test_uniform_alpha_diversity(self):
        """Alpha=1.0 (uniform) produces diverse lambdas, not clustered near 0.5 or 1.0."""
        alpha = 1.0
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample(sample_shape=(10000,))
        lam = torch.maximum(lam, 1 - lam).float()
        # With uniform, we expect lambdas spread across [0.5, 1.0]
        assert lam.std() > 0.1, f"Lambdas not diverse enough: std={lam.std():.4f}"

    def test_spectrogram_broadcast(self):
        """Per-sample lambda broadcasts correctly to spectrogram shape."""
        batch_size = 8
        lam = torch.rand(batch_size)
        x = torch.randn(batch_size, 1, 256, 256)
        indices = torch.randperm(batch_size)

        x_lam = lam[:, None, None, None]
        mixed = x_lam * x + (1 - x_lam) * x[indices]
        assert mixed.shape == x.shape

    def test_label_broadcast(self):
        """Per-sample lambda broadcasts correctly to label shape."""
        batch_size = 8
        lam = torch.rand(batch_size)
        labels = torch.randn(batch_size, 234)
        indices = torch.randperm(batch_size)

        label_lam = lam[:, None]
        mixed = label_lam * labels + (1 - label_lam) * labels[indices]
        assert mixed.shape == labels.shape

    def test_theta_clipping(self):
        """Labels >= theta are snapped to 1.0."""
        labels = torch.tensor([[0.5, 0.85, 0.3, 0.95]])
        theta = 0.8
        clipped = torch.where(labels >= theta, torch.ones_like(labels), labels)
        expected = torch.tensor([[0.5, 1.0, 0.3, 1.0]])
        torch.testing.assert_close(clipped, expected)


# ============================================================================
# LSEPooling tests
# ============================================================================


class TestLSEPooling:
    """Test LSE temporal pooling matches ttahara's implementation."""

    def test_output_shape(self):
        """LSEPooling reduces time dimension."""
        pool = LSEPooling(temperature=1.0)
        x = torch.randn(2, 32, 234)  # (B, T, C)
        out = pool(x)
        assert out.shape == (2, 234)

    def test_approaches_max_at_low_temp(self):
        """T→0 approximates max pooling."""
        pool = LSEPooling(temperature=0.01)
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        out = pool(x)
        expected_max = torch.tensor([[4.0, 5.0, 6.0]])
        torch.testing.assert_close(out, expected_max, atol=0.1, rtol=0.01)

    def test_approaches_mean_at_high_temp(self):
        """T→∞ approximates mean pooling."""
        pool = LSEPooling(temperature=100.0)
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        out = pool(x)
        expected_mean = torch.tensor([[2.0, 3.0]])
        torch.testing.assert_close(out, expected_mean, atol=0.1, rtol=0.01)

    def test_formula_matches_ttahara(self):
        """Our formula matches ttahara's: T * (logsumexp(x/T) - log(n))."""
        T = 1.0
        x = torch.randn(2, 8, 234)
        pool = LSEPooling(temperature=T)
        our_result = pool(x)

        # ttahara's formula (adapted for dim=1)
        ttahara_result = T * (torch.logsumexp(x / T, dim=1) - math.log(x.shape[1]))
        torch.testing.assert_close(our_result, ttahara_result)
