"""Student model architectures for distillation.

These lightweight sequence models learn to contextualize static embeddings
by predicting the full BEATs transformer's output.

Reference: user/anthony/experiments/003-model-distillation/research/student-implementations.md
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class ConvNeXt1DBlock(nn.Module):
    """Single ConvNeXt-style 1D block with dilated depthwise convolution."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        dilation: int = 1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        residual = x
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, dim)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


class ConvNeXt1DStudent(nn.Module):
    """ConvNeXt-style 1D architecture with dilated depthwise convolutions.

    Modern CNN combining best practices from ConvNeXt, MobileNet, and WaveNet.
    Uses dilated convolutions for exponential receptive field growth.

    Args:
        in_dim: Input embedding dimension (768 for BEATs)
        hidden_dim: Hidden dimension for ConvNeXt blocks
        out_dim: Output embedding dimension (768 to match teacher)
        num_blocks: Number of ConvNeXt blocks
        kernel_size: Kernel size for depthwise convolutions
        expansion: Expansion ratio for inverted bottleneck
        dilations: Dilation rates for each block (cycles if fewer than num_blocks)
        drop_path: Drop path rate for stochastic depth
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 768,
        num_blocks: int = 6,
        kernel_size: int = 7,
        expansion: int = 4,
        dilations: list[int] | None = None,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ConvNeXt1DBlock(
                    dim=hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    dilation=dilations[i % len(dilations)],
                    drop_path=drop_path * i / num_blocks,
                )
                for i in range(num_blocks)
            ]
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Static embeddings (batch, seq_len, in_dim)
        Returns:
            Contextualized embeddings (batch, seq_len, out_dim)
        """
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(self.proj_out(x))


class MixerBlock(nn.Module):
    """Single MLP-Mixer block with token mixing and channel mixing."""

    def __init__(
        self,
        dim: int,
        seq_len: int,
        token_expansion: float = 0.5,
        channel_expansion: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, int(seq_len * token_expansion)),
            nn.GELU(),
            nn.Linear(int(seq_len * token_expansion), seq_len),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * channel_expansion),
            nn.GELU(),
            nn.Linear(dim * channel_expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        # Token mixing: transpose to (batch, dim, seq_len), mix, transpose back
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        # Channel mixing
        x = x + self.channel_mix(self.norm2(x))
        return x


class MLPMixerStudent(nn.Module):
    """MLP-Mixer for sequence contextualization.

    Pure MLP architecture with token mixing (across sequence) and
    channel mixing (across features).

    Args:
        in_dim: Input embedding dimension (768 for BEATs)
        hidden_dim: Hidden dimension for Mixer blocks
        out_dim: Output embedding dimension (768 to match teacher)
        seq_len: Sequence length (248 patches for 5s audio)
        num_blocks: Number of Mixer blocks
        token_expansion: Expansion ratio for token mixing MLP
        channel_expansion: Expansion ratio for channel mixing MLP
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 768,
        seq_len: int = 248,
        num_blocks: int = 4,
        token_expansion: float = 0.5,
        channel_expansion: int = 4,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(hidden_dim, seq_len, token_expansion, channel_expansion)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Static embeddings (batch, seq_len, in_dim)
        Returns:
            Contextualized embeddings (batch, seq_len, out_dim)
        """
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(self.norm(x))


class TinyTransformerStudent(nn.Module):
    """Small transformer for baseline comparison.

    Uses PyTorch's TransformerEncoder with pre-norm architecture.
    O(n^2) attention complexity, but manageable for 248 tokens.

    Args:
        in_dim: Input embedding dimension (768 for BEATs)
        hidden_dim: Model dimension for transformer
        out_dim: Output embedding dimension (768 to match teacher)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Static embeddings (batch, seq_len, in_dim)
        Returns:
            Contextualized embeddings (batch, seq_len, out_dim)
        """
        x = self.proj_in(x)
        x = self.transformer(x)
        return self.proj_out(x)


def get_student_model(name: str, **kwargs) -> nn.Module:
    """Factory function to create student models.

    Args:
        name: Model name (convnext, mixer, transformer)
        **kwargs: Model-specific arguments

    Returns:
        Student model instance
    """
    models = {
        "convnext": ConvNeXt1DStudent,
        "convnext-tiny": lambda **kw: ConvNeXt1DStudent(
            hidden_dim=192, num_blocks=4, dilations=[1, 2, 4, 8], **kw
        ),
        "convnext-small": lambda **kw: ConvNeXt1DStudent(
            hidden_dim=256, num_blocks=6, dilations=[1, 2, 4, 8, 16, 32], **kw
        ),
        "convnext-base": lambda **kw: ConvNeXt1DStudent(
            hidden_dim=384, num_blocks=8, dilations=[1, 2, 4, 8, 16, 32, 64, 128], **kw
        ),
        "mixer": MLPMixerStudent,
        "mixer-tiny": lambda **kw: MLPMixerStudent(
            hidden_dim=256, num_blocks=2, **kw
        ),
        "mixer-small": lambda **kw: MLPMixerStudent(
            hidden_dim=384, num_blocks=4, **kw
        ),
        "transformer": TinyTransformerStudent,
        "transformer-tiny": lambda **kw: TinyTransformerStudent(
            hidden_dim=256, num_layers=2, num_heads=4, **kw
        ),
        "transformer-small": lambda **kw: TinyTransformerStudent(
            hidden_dim=384, num_layers=4, num_heads=6, **kw
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
