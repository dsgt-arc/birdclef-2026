"""Tests for mamba-ssm CUDA kernel integration.

These tests verify:
1. Module imports and availability detection work correctly
2. PyTorch fallback backend works without mamba-ssm
3. CUDA backend works when mamba-ssm is available
4. Both backends produce numerically equivalent outputs
"""

import time

import numpy as np
import pytest
import torch

from ssamba_hf import (
    MAMBA_SSM_AVAILABLE,
    SSAMBAConfig,
    SSAMBAModel,
    is_cuda_kernel_available,
)


# Skip markers
requires_mamba_ssm = pytest.mark.skipif(
    not MAMBA_SSM_AVAILABLE,
    reason="mamba-ssm not installed",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@pytest.fixture
def sample_spectrogram():
    """Create a reproducible test spectrogram."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    return torch.from_numpy(spec)


@pytest.fixture
def small_config():
    """Create a small model config for testing."""
    return SSAMBAConfig.small()


class TestAvailabilityDetection:
    """Test CUDA kernel availability detection."""

    def test_mamba_ssm_available_is_bool(self):
        """MAMBA_SSM_AVAILABLE should be a boolean."""
        assert isinstance(MAMBA_SSM_AVAILABLE, bool)

    def test_is_cuda_kernel_available_returns_bool(self):
        """is_cuda_kernel_available() should return a boolean."""
        result = is_cuda_kernel_available()
        assert isinstance(result, bool)

    def test_availability_consistency(self):
        """is_cuda_kernel_available() should match MAMBA_SSM_AVAILABLE."""
        assert is_cuda_kernel_available() == MAMBA_SSM_AVAILABLE


class TestConfigIntegration:
    """Test use_cuda_kernels config option."""

    def test_config_default_is_none(self):
        """Default use_cuda_kernels should be None (auto-detect)."""
        config = SSAMBAConfig()
        assert config.use_cuda_kernels is None

    def test_config_accepts_false(self):
        """Config should accept use_cuda_kernels=False."""
        config = SSAMBAConfig(use_cuda_kernels=False)
        assert config.use_cuda_kernels is False

    def test_config_accepts_true(self):
        """Config should accept use_cuda_kernels=True."""
        config = SSAMBAConfig(use_cuda_kernels=True)
        assert config.use_cuda_kernels is True

    def test_config_passed_to_mixer(self, small_config):
        """Config use_cuda_kernels should be passed to BidirectionalMamba."""
        config = SSAMBAConfig.small(use_cuda_kernels=False)
        model = SSAMBAModel(config)

        # Check first layer's mixer
        mixer = model.layers.layers[0].mixer
        assert mixer._use_cuda_kernels is False


class TestPyTorchBackend:
    """Test pure PyTorch backend (always available)."""

    def test_model_forward_cpu(self, sample_spectrogram, small_config):
        """Model should run on CPU with PyTorch backend."""
        config = SSAMBAConfig.small(use_cuda_kernels=False)
        model = SSAMBAModel(config)
        model.eval()

        with torch.no_grad():
            embeddings = model.get_embeddings(sample_spectrogram)

        assert embeddings.shape == (1, config.embed_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_model_forward_deterministic(self, sample_spectrogram):
        """Same input should produce same output."""
        config = SSAMBAConfig.small(use_cuda_kernels=False)
        model = SSAMBAModel(config)
        model.eval()

        with torch.no_grad():
            emb1 = model.get_embeddings(sample_spectrogram)
            emb2 = model.get_embeddings(sample_spectrogram)

        assert torch.allclose(emb1, emb2)

    @requires_cuda
    def test_pytorch_backend_on_cuda(self, sample_spectrogram):
        """PyTorch backend should work on CUDA when forced."""
        config = SSAMBAConfig.small(use_cuda_kernels=False)
        model = SSAMBAModel(config).cuda()
        model.eval()

        spec_cuda = sample_spectrogram.cuda()

        with torch.no_grad():
            embeddings = model.get_embeddings(spec_cuda)

        assert embeddings.device.type == "cuda"
        assert embeddings.shape == (1, config.embed_dim)
        assert not torch.isnan(embeddings).any()


class TestCUDABackend:
    """Test mamba-ssm CUDA kernel backend."""

    @requires_mamba_ssm
    @requires_cuda
    def test_cuda_backend_runs(self, sample_spectrogram):
        """CUDA backend should run when mamba-ssm is available."""
        config = SSAMBAConfig.small(use_cuda_kernels=True)
        model = SSAMBAModel(config).cuda()
        model.eval()

        spec_cuda = sample_spectrogram.cuda()

        with torch.no_grad():
            embeddings = model.get_embeddings(spec_cuda)

        assert embeddings.device.type == "cuda"
        assert embeddings.shape == (1, config.embed_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    @requires_mamba_ssm
    @requires_cuda
    def test_auto_detection_uses_cuda(self, sample_spectrogram):
        """Auto mode (None) should use CUDA kernels when available on GPU."""
        config = SSAMBAConfig.small(use_cuda_kernels=None)
        model = SSAMBAModel(config).cuda()
        model.eval()

        spec_cuda = sample_spectrogram.cuda()

        # The model should auto-detect and use CUDA kernels
        mixer = model.layers.layers[0].mixer
        assert mixer._should_use_cuda_kernels(spec_cuda) is True

    def test_require_cuda_raises_without_mamba_ssm(self, sample_spectrogram):
        """use_cuda_kernels=True should raise if mamba-ssm not available."""
        if MAMBA_SSM_AVAILABLE:
            pytest.skip("mamba-ssm is available, cannot test missing case")

        config = SSAMBAConfig.small(use_cuda_kernels=True)
        model = SSAMBAModel(config)
        model.eval()

        with pytest.raises(RuntimeError, match="mamba-ssm not installed"):
            with torch.no_grad():
                model.get_embeddings(sample_spectrogram)

    @requires_mamba_ssm
    def test_require_cuda_raises_on_cpu(self, sample_spectrogram):
        """use_cuda_kernels=True should raise if tensor is on CPU."""
        config = SSAMBAConfig.small(use_cuda_kernels=True)
        model = SSAMBAModel(config)  # CPU model
        model.eval()

        with pytest.raises(RuntimeError, match="not on CUDA"):
            with torch.no_grad():
                model.get_embeddings(sample_spectrogram)  # CPU tensor


class TestNumericalEquivalence:
    """Test that CUDA and PyTorch backends produce equivalent outputs."""

    @requires_mamba_ssm
    @requires_cuda
    def test_backends_match(self, sample_spectrogram):
        """CUDA and PyTorch backends should produce similar outputs."""
        # PyTorch backend (CPU)
        config_pytorch = SSAMBAConfig.small(use_cuda_kernels=False)
        model_pytorch = SSAMBAModel(config_pytorch)
        model_pytorch.eval()

        with torch.no_grad():
            emb_pytorch = model_pytorch.get_embeddings(sample_spectrogram)

        # CUDA backend
        config_cuda = SSAMBAConfig.small(use_cuda_kernels=True)
        model_cuda = SSAMBAModel(config_cuda).cuda()
        # Copy weights from PyTorch model
        model_cuda.load_state_dict(model_pytorch.state_dict())
        model_cuda.eval()

        spec_cuda = sample_spectrogram.cuda()

        with torch.no_grad():
            emb_cuda = model_cuda.get_embeddings(spec_cuda)

        emb_cuda_cpu = emb_cuda.cpu()

        # Compute similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            emb_pytorch, emb_cuda_cpu
        ).item()

        # Should be very similar (allowing for numerical precision differences)
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.4f} is too low"

    @requires_mamba_ssm
    @requires_cuda
    def test_backends_match_high_precision(self, sample_spectrogram):
        """CUDA and PyTorch backends should be nearly identical."""
        # Create model with shared weights
        config = SSAMBAConfig.small(use_cuda_kernels=False)
        model = SSAMBAModel(config)
        model.eval()

        # Get PyTorch output
        with torch.no_grad():
            emb_pytorch = model.get_embeddings(sample_spectrogram)

        # Move to CUDA and enable CUDA kernels
        model = model.cuda()
        for layer in model.layers.layers:
            layer.mixer._use_cuda_kernels = True

        spec_cuda = sample_spectrogram.cuda()

        with torch.no_grad():
            emb_cuda = model.get_embeddings(spec_cuda)

        emb_cuda_cpu = emb_cuda.cpu()

        cos_sim = torch.nn.functional.cosine_similarity(
            emb_pytorch, emb_cuda_cpu
        ).item()
        max_diff = (emb_pytorch - emb_cuda_cpu).abs().max().item()

        # Report metrics
        print(f"\nCosine similarity: {cos_sim:.6f}")
        print(f"Max absolute diff: {max_diff:.2e}")

        # High precision check
        assert cos_sim > 0.999, f"Expected cosine > 0.999, got {cos_sim:.6f}"


class TestBenchmark:
    """Benchmark tests (informational, not strict assertions)."""

    @requires_mamba_ssm
    @requires_cuda
    def test_cuda_faster_than_pytorch(self, sample_spectrogram):
        """CUDA backend should be faster than PyTorch backend."""
        # PyTorch backend timing (CPU)
        config_pytorch = SSAMBAConfig.small(use_cuda_kernels=False)
        model_pytorch = SSAMBAModel(config_pytorch)
        model_pytorch.eval()

        with torch.no_grad():
            # Warmup
            _ = model_pytorch.get_embeddings(sample_spectrogram)

            t0 = time.perf_counter()
            _ = model_pytorch.get_embeddings(sample_spectrogram)
            t_pytorch = time.perf_counter() - t0

        # CUDA backend timing
        config_cuda = SSAMBAConfig.small(use_cuda_kernels=True)
        model_cuda = SSAMBAModel(config_cuda).cuda()
        model_cuda.eval()

        spec_cuda = sample_spectrogram.cuda()

        with torch.no_grad():
            # Warmup
            _ = model_cuda.get_embeddings(spec_cuda)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model_cuda.get_embeddings(spec_cuda)
            torch.cuda.synchronize()
            t_cuda = time.perf_counter() - t0

        speedup = t_pytorch / t_cuda if t_cuda > 0 else float("inf")

        print(f"\nPyTorch (CPU): {t_pytorch:.3f}s")
        print(f"CUDA kernel:   {t_cuda:.3f}s")
        print(f"Speedup:       {speedup:.1f}x")

        # CUDA should be faster (at least 2x for this simple test)
        # Note: This is a soft check - actual speedup depends on hardware
        assert speedup > 1.0, "CUDA should be faster than CPU PyTorch"
