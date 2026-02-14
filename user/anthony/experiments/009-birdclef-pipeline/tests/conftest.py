"""Shared test fixtures and skip markers."""

import pytest
import torch


@pytest.fixture
def device():
    """Return best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@pytest.fixture
def tiny_config():
    """SSAMBA Tiny config for testing."""
    from ssamba_hf.config import SSAMBAConfig
    return SSAMBAConfig(
        img_size=(128, 1000),
        patch_size=(128, 1),
        embed_dim=192,
        depth=4,  # Reduced depth for fast tests
    )


@pytest.fixture
def mel_spec_layer():
    """nnAudio MelSpectrogram layer for testing."""
    from nnAudio.features import MelSpectrogram
    return MelSpectrogram(
        sr=32000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,  # 32000 * 10 / 1000 = 320
        n_mels=128,
        fmin=50,
        fmax=14000,
        trainable_mel=False,
        trainable_STFT=False,
    )
