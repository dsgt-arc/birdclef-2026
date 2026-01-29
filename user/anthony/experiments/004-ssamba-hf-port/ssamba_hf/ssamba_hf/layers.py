"""Basic layers for SSAMBA model."""

import torch
import torch.nn as nn
from typing import Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467

    Note: RMSNorm has no bias parameter. The SSAMBA small-400 checkpoint was
    pretrained with rms_norm=false (nn.LayerNorm), so norm layers have trained
    biases. When loading such checkpoints, use rms_norm=False in the config to
    select nn.LayerNorm instead.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class AudioPatchEmbed(nn.Module):
    """Audio Patch Embedding.

    Converts spectrogram to patch embeddings using a Conv2d layer.

    Input: (batch, 1, freq_bins, time_frames)
    Output: (batch, num_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 1024),
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_chans, freq_bins, time_frames)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, E, H', W') -> (B, E, N) -> (B, N, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Reference: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
