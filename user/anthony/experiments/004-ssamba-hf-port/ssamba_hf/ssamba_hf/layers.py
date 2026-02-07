"""Basic layers for SSAMBA model."""

import torch
import torch.nn as nn
from typing import Tuple, Union


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
    Supports overlapping patches via separate fstride/tstride parameters.

    Input: (batch, 1, freq_bins, time_frames)
    Output: (batch, num_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 1024),
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        fstride: int = None,
        tstride: int = None,
    ):
        super().__init__()
        self.img_size = img_size
        # Normalize patch_size to (freq, time) tuple
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)
        self.fstride = fstride if fstride is not None else self.patch_size[0]
        self.tstride = tstride if tstride is not None else self.patch_size[1]

        # Calculate num_patches based on stride (supports overlapping)
        self.f_dim = (img_size[0] - self.patch_size[0]) // self.fstride + 1
        self.t_dim = (img_size[1] - self.patch_size[1]) // self.tstride + 1
        self.num_patches = self.f_dim * self.t_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=(self.fstride, self.tstride),
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
