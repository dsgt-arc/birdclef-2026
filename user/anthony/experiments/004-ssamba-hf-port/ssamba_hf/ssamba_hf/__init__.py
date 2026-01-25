"""SSAMBA HuggingFace Port - Pure PyTorch implementation of SSAMBA.

This package provides a CPU-compatible implementation of SSAMBA (Self-Supervised
Audio Mamba) using HuggingFace's transformers library instead of the CUDA-only
mamba_ssm kernels.

Usage:
    from ssamba_hf import SSAMBAModel, SSAMBAConfig, load_from_ssamba_checkpoint

    # Load from original SSAMBA checkpoint
    model, config = load_from_ssamba_checkpoint("path/to/amba_small_400.pth")

    # Extract embeddings
    spectrogram = torch.randn(1, 1024, 128)  # (batch, time_frames, freq_bins)
    embeddings = model.get_embeddings(spectrogram)  # (batch, 384)
"""

from .config import SSAMBAConfig
from .layers import RMSNorm, AudioPatchEmbed, DropPath
from .mamba import BidirectionalMambaStack, MambaBlock, BidirectionalMamba
from .model import SSAMBAModel
from .convert import convert_ssamba_weights, load_from_ssamba_checkpoint, analyze_checkpoint

__all__ = [
    # Config
    "SSAMBAConfig",
    # Layers
    "RMSNorm",
    "AudioPatchEmbed",
    "DropPath",
    # Mamba
    "BidirectionalMambaStack",
    "MambaBlock",
    "BidirectionalMamba",
    # Model
    "SSAMBAModel",
    # Conversion
    "convert_ssamba_weights",
    "load_from_ssamba_checkpoint",
    "analyze_checkpoint",
]

__version__ = "0.1.0"
