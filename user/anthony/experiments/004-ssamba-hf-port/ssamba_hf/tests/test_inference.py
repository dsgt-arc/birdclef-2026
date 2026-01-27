"""Tests for long audio inference utilities.

These tests verify:
1. Windowed inference produces valid embeddings for long audio
2. Short audio passthrough works correctly
3. Different aggregation modes produce expected results
4. Edge cases (exact window size, slightly over) are handled
"""

import numpy as np
import pytest
import torch

from ssamba_hf import (
    SSAMBAConfig,
    SSAMBAModel,
    get_embeddings_long,
    estimate_num_windows,
    pad_to_window_size,
)
from ssamba_hf.inference import _compute_window_positions


@pytest.fixture
def small_config():
    """Create a small model config for testing."""
    return SSAMBAConfig.small(use_cuda_kernels=False)


@pytest.fixture
def model(small_config):
    """Create a model for testing."""
    model = SSAMBAModel(small_config)
    model.eval()
    return model


@pytest.fixture
def short_spectrogram():
    """Create a short spectrogram (< 1024 frames)."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((1, 500, 128)).astype(np.float32)
    return torch.from_numpy(spec)


@pytest.fixture
def exact_spectrogram():
    """Create a spectrogram with exactly 1024 frames."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    return torch.from_numpy(spec)


@pytest.fixture
def long_spectrogram():
    """Create a long spectrogram (> 1024 frames)."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((1, 3000, 128)).astype(np.float32)
    return torch.from_numpy(spec)


@pytest.fixture
def batch_long_spectrogram():
    """Create a batch of long spectrograms."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((4, 2500, 128)).astype(np.float32)
    return torch.from_numpy(spec)


class TestWindowPositions:
    """Test window position computation."""

    def test_basic_positions(self):
        """Basic window positions should be computed correctly."""
        positions = _compute_window_positions(
            num_frames=3000, window_size=1024, hop_size=512
        )

        assert positions[0] == 0
        assert positions[1] == 512
        assert positions[2] == 1024
        assert positions[3] == 1536
        # Last position should cover the end
        assert positions[-1] == 3000 - 1024  # 1976

    def test_exact_fit(self):
        """Exact fit should produce expected positions."""
        # 1024 + 512 + 512 = 2048 -> 3 windows
        positions = _compute_window_positions(
            num_frames=2048, window_size=1024, hop_size=512
        )

        assert positions == [0, 512, 1024]

    def test_single_window(self):
        """Single window should work."""
        positions = _compute_window_positions(
            num_frames=1024, window_size=1024, hop_size=512
        )

        assert positions == [0]

    def test_slightly_over_one_window(self):
        """Slightly over one window should produce two positions."""
        positions = _compute_window_positions(
            num_frames=1025, window_size=1024, hop_size=512
        )

        # First at 0, second should cover the end
        assert 0 in positions
        assert 1 in positions  # 1025 - 1024 = 1

    def test_covers_end(self):
        """Last window should always cover the end of the sequence."""
        for num_frames in [1500, 2000, 2500, 3000, 3500]:
            positions = _compute_window_positions(
                num_frames=num_frames, window_size=1024, hop_size=512
            )

            # Last window should end at or after num_frames
            assert positions[-1] + 1024 >= num_frames
            # And start at a valid position
            assert positions[-1] >= 0


class TestLongAudioInference:
    """Test windowed inference for long audio."""

    def test_long_audio_basic(self, model, long_spectrogram):
        """Basic long audio inference should work."""
        embeddings = get_embeddings_long(model, long_spectrogram)

        assert embeddings.shape == (1, 384)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_exact_window_size(self, model, exact_spectrogram):
        """Exact window size should process correctly."""
        embeddings = get_embeddings_long(model, exact_spectrogram)

        assert embeddings.shape == (1, 384)
        assert not torch.isnan(embeddings).any()

    def test_short_audio_raises(self, model, short_spectrogram):
        """Short audio should raise ValueError."""
        with pytest.raises(ValueError, match="frames"):
            get_embeddings_long(model, short_spectrogram)

    def test_batch_long_audio(self, model, batch_long_spectrogram):
        """Batched long audio should work."""
        embeddings = get_embeddings_long(model, batch_long_spectrogram)

        assert embeddings.shape == (4, 384)
        assert not torch.isnan(embeddings).any()

    def test_deterministic(self, model, long_spectrogram):
        """Same input should produce same output."""
        emb1 = get_embeddings_long(model, long_spectrogram)
        emb2 = get_embeddings_long(model, long_spectrogram)

        assert torch.allclose(emb1, emb2)

    def test_custom_window_size(self, model, long_spectrogram):
        """Custom window/hop sizes should work."""
        embeddings = get_embeddings_long(
            model, long_spectrogram, window_size=1024, hop_size=256
        )

        assert embeddings.shape == (1, 384)


class TestAggregationModes:
    """Test different aggregation strategies."""

    def test_mean_aggregation(self, model, long_spectrogram):
        """Mean aggregation should work."""
        embeddings = get_embeddings_long(
            model, long_spectrogram, aggregation="mean"
        )

        assert embeddings.shape == (1, 384)

    def test_max_aggregation(self, model, long_spectrogram):
        """Max aggregation should work."""
        embeddings = get_embeddings_long(
            model, long_spectrogram, aggregation="max"
        )

        assert embeddings.shape == (1, 384)

    def test_last_aggregation(self, model, long_spectrogram):
        """Last aggregation should work."""
        embeddings = get_embeddings_long(
            model, long_spectrogram, aggregation="last"
        )

        assert embeddings.shape == (1, 384)

    def test_invalid_aggregation(self, model, long_spectrogram):
        """Invalid aggregation should raise."""
        with pytest.raises(ValueError, match="aggregation"):
            get_embeddings_long(model, long_spectrogram, aggregation="invalid")

    def test_aggregations_differ(self, model, long_spectrogram):
        """Different aggregations should produce different results."""
        emb_mean = get_embeddings_long(model, long_spectrogram, aggregation="mean")
        emb_max = get_embeddings_long(model, long_spectrogram, aggregation="max")
        emb_last = get_embeddings_long(model, long_spectrogram, aggregation="last")

        # They should all be different
        assert not torch.allclose(emb_mean, emb_max)
        assert not torch.allclose(emb_mean, emb_last)


class TestNormalization:
    """Test output normalization."""

    def test_normalize_true(self, model, long_spectrogram):
        """Normalization should produce unit vectors."""
        embeddings = get_embeddings_long(model, long_spectrogram, normalize=True)

        # Check L2 norm is ~1
        norm = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_normalize_false(self, model, long_spectrogram):
        """Without normalization, vectors may not be unit length."""
        embeddings = get_embeddings_long(model, long_spectrogram, normalize=False)

        # Norm can be anything
        norm = torch.norm(embeddings, p=2, dim=-1)
        # Just check it's not NaN/Inf
        assert not torch.isnan(norm).any()


class TestEstimateNumWindows:
    """Test window count estimation."""

    def test_basic_estimate(self):
        """Basic estimate should work."""
        num = estimate_num_windows(3000, window_size=1024, hop_size=512)
        assert num == 5  # [0, 512, 1024, 1536, 1976]

    def test_short_audio(self):
        """Short audio should return 0."""
        num = estimate_num_windows(500, window_size=1024, hop_size=512)
        assert num == 0

    def test_exact_window(self):
        """Exact window should return 1."""
        num = estimate_num_windows(1024, window_size=1024, hop_size=512)
        assert num == 1

    def test_matches_actual_processing(self, model, long_spectrogram):
        """Estimate should match actual window count."""
        num_frames = long_spectrogram.shape[1]
        estimated = estimate_num_windows(num_frames, window_size=1024, hop_size=512)
        actual = len(_compute_window_positions(num_frames, 1024, 512))

        assert estimated == actual


class TestPadToWindowSize:
    """Test padding utilities."""

    def test_short_padded_constant(self, short_spectrogram):
        """Short spectrogram should be padded with zeros."""
        padded = pad_to_window_size(short_spectrogram, window_size=1024, pad_mode="constant")

        assert padded.shape == (1, 1024, 128)
        # Original content preserved
        assert torch.allclose(padded[:, :500, :], short_spectrogram)
        # Padding is zeros
        assert (padded[:, 500:, :] == 0).all()

    def test_short_padded_replicate(self, short_spectrogram):
        """Replicate padding should repeat edge values."""
        padded = pad_to_window_size(short_spectrogram, window_size=1024, pad_mode="replicate")

        assert padded.shape == (1, 1024, 128)
        # Original content preserved
        assert torch.allclose(padded[:, :500, :], short_spectrogram)
        # Last row should be replicated
        assert torch.allclose(padded[:, 500, :], padded[:, 999, :])

    def test_short_padded_reflect(self, short_spectrogram):
        """Reflect padding should mirror edge values."""
        padded = pad_to_window_size(short_spectrogram, window_size=1024, pad_mode="reflect")

        assert padded.shape == (1, 1024, 128)
        # Original content preserved
        assert torch.allclose(padded[:, :500, :], short_spectrogram)

    def test_long_audio_unchanged(self, long_spectrogram):
        """Long audio should be unchanged."""
        padded = pad_to_window_size(long_spectrogram, window_size=1024)

        assert padded.shape == long_spectrogram.shape
        assert torch.equal(padded, long_spectrogram)

    def test_exact_window_unchanged(self, exact_spectrogram):
        """Exact window size should be unchanged."""
        padded = pad_to_window_size(exact_spectrogram, window_size=1024)

        assert padded.shape == exact_spectrogram.shape
        assert torch.equal(padded, exact_spectrogram)

    def test_invalid_pad_mode(self, short_spectrogram):
        """Invalid pad mode should raise."""
        with pytest.raises(ValueError, match="pad_mode"):
            pad_to_window_size(short_spectrogram, pad_mode="invalid")


class TestIntegration:
    """Integration tests combining padding and inference."""

    def test_pad_then_process(self, model, short_spectrogram):
        """Padded short audio should be processable."""
        padded = pad_to_window_size(short_spectrogram, window_size=1024)
        embeddings = model.get_embeddings(padded)

        assert embeddings.shape == (1, 384)
        assert not torch.isnan(embeddings).any()

    def test_long_audio_matches_direct(self, model, exact_spectrogram):
        """get_embeddings_long should match direct call for exact window."""
        emb_direct = model.get_embeddings(exact_spectrogram)
        emb_long = get_embeddings_long(model, exact_spectrogram)

        assert torch.allclose(emb_direct, emb_long)
