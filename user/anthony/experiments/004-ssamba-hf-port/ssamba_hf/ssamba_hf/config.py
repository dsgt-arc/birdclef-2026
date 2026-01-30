"""SSAMBA model configuration."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SSAMBAConfig:
    """Configuration for SSAMBA model.

    Attributes:
        embed_dim: Embedding dimension (192/384/768 for tiny/small/base)
        depth: Number of Mamba blocks (24 for all sizes)
        img_size: Input size as (freq_bins, time_frames)
        patch_size: Patch size for patch embedding
        in_chans: Number of input channels (1 for spectrogram)
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        use_cls_token: Whether to use a CLS token
        rms_norm: Whether to use RMSNorm (vs LayerNorm)
        residual_in_fp32: Whether to keep residual in fp32
        drop_path_rate: Drop path rate for regularization
    """

    # Model size
    embed_dim: int = 384  # 192/384/768 for tiny/small/base
    depth: int = 24  # Number of Mamba blocks

    # Patch embedding
    img_size: Tuple[int, int] = (128, 1024)  # (freq_bins, time_frames)
    patch_size: int = 16
    fstride: Optional[int] = None  # Frequency stride (default: patch_size)
    tstride: Optional[int] = None  # Time stride (default: patch_size)
    in_chans: int = 1

    # Mamba SSM parameters
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor for inner dim

    # Token configuration
    use_cls_token: bool = True
    use_middle_cls_token: bool = False  # CLS token prepended (pretraining uses false)

    # Normalization
    rms_norm: bool = False  # Use nn.LayerNorm (pretraining uses rms_norm=false)
    residual_in_fp32: bool = False  # Pretraining uses residual_in_fp32=false

    # Regularization
    drop_path_rate: float = 0.0

    # CUDA kernel optimization
    use_cuda_kernels: Optional[bool] = None  # None=auto, True=require, False=disable

    @property
    def num_patches(self) -> int:
        """Calculate number of patches."""
        f_patches = self.img_size[0] // self.patch_size
        t_patches = self.img_size[1] // self.patch_size
        return f_patches * t_patches

    @property
    def d_inner(self) -> int:
        """Calculate inner dimension for Mamba."""
        return self.embed_dim * self.expand

    @classmethod
    def tiny(cls, **kwargs) -> "SSAMBAConfig":
        """Create tiny model config (embed_dim=192)."""
        return cls(embed_dim=192, depth=24, **kwargs)

    @classmethod
    def small(cls, **kwargs) -> "SSAMBAConfig":
        """Create small model config (embed_dim=384)."""
        return cls(embed_dim=384, depth=24, **kwargs)

    @classmethod
    def base(cls, **kwargs) -> "SSAMBAConfig":
        """Create base model config (embed_dim=768)."""
        return cls(embed_dim=768, depth=24, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "SSAMBAConfig":
        """Infer config from checkpoint."""
        import re
        import torch

        sd = torch.load(checkpoint_path, map_location=device)

        # Extract config from checkpoint keys
        patch_weight = sd.get(
            "module.v.patch_embed.proj.weight",
            sd.get("v.patch_embed.proj.weight")
        )
        pos_embed = sd.get("module.v.pos_embed", sd.get("v.pos_embed"))
        input_fdim = sd.get("module.p_input_fdim", sd.get("p_input_fdim"))
        input_tdim = sd.get("module.p_input_tdim", sd.get("p_input_tdim"))

        if patch_weight is None or pos_embed is None:
            raise ValueError("Cannot infer config from checkpoint")

        patch_size = patch_weight.shape[2]  # Assuming square patches
        embed_dim = pos_embed.shape[2]

        # Input dimensions
        if hasattr(input_fdim, "item"):
            input_fdim = input_fdim.item()
            input_tdim = input_tdim.item()
        else:
            input_fdim = int(input_fdim)
            input_tdim = int(input_tdim)

        # Infer depth from layer indices in checkpoint
        layer_indices = set()
        for key in sd.keys():
            match = re.search(r"layers\.(\d+)\.", key)
            if match:
                layer_indices.add(int(match.group(1)))

        # The number of layers is max_index + 1
        depth = max(layer_indices) + 1 if layer_indices else 24
        print(f"Inferred depth={depth} from checkpoint (found layer indices: {sorted(layer_indices)[:5]}...)")

        # Auto-detect norm type: LayerNorm has bias, RMSNorm does not
        has_norm_bias = any(k.endswith("norm.bias") for k in sd.keys())
        rms_norm = not has_norm_bias
        print(f"Inferred rms_norm={rms_norm} from checkpoint (norm.bias {'absent' if rms_norm else 'present'})")

        return cls(
            embed_dim=embed_dim,
            depth=depth,
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size,
            rms_norm=rms_norm,
        )
