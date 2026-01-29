"""Extract embeddings using the original SSAMBA implementation.

Requires:
- mamba_ssm and causal_conv1d installed
- VisionMamba from Vim repository in PYTHONPATH
- CUDA GPU available
"""

import argparse
import torch
import torch.nn as nn
import numpy as np


# Provide RMSNorm fallback for when triton ops aren't available
class RMSNorm(nn.Module):
    """Pure PyTorch RMSNorm implementation."""
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


# Provide RMSNorm fallback if triton ops aren't available
# The custom mamba_ssm from Vim/mamba-1p1p1 should be installed (with bidirectional support)
import sys
from types import ModuleType

# Check if triton layernorm is available, otherwise provide fallback
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm as TritonRMSNorm
    print(f"Using triton RMSNorm: {TritonRMSNorm}")
except ImportError:
    print("Triton RMSNorm not available, providing PyTorch fallback")
    # Create fake modules with our RMSNorm fallback
    import mamba_ssm.ops

    if "mamba_ssm.ops.triton" not in sys.modules:
        fake_triton = ModuleType("mamba_ssm.ops.triton")
        sys.modules["mamba_ssm.ops.triton"] = fake_triton
        mamba_ssm.ops.triton = fake_triton

    # Create fake layernorm modules with our RMSNorm
    # Note: different files use different module names:
    #   - models_mamba.py imports from "layer_norm" (with underscore)
    #   - both_models.py imports from "layernorm" (no underscore)
    fake_layer_norm = ModuleType("mamba_ssm.ops.triton.layer_norm")
    fake_layer_norm.RMSNorm = RMSNorm
    fake_layer_norm.layer_norm_fn = None
    fake_layer_norm.rms_norm_fn = None

    fake_layernorm = ModuleType("mamba_ssm.ops.triton.layernorm")
    fake_layernorm.RMSNorm = RMSNorm
    fake_layernorm.layer_norm_fn = None
    fake_layernorm.rms_norm_fn = None

    sys.modules["mamba_ssm.ops.triton.layer_norm"] = fake_layer_norm
    sys.modules["mamba_ssm.ops.triton.layernorm"] = fake_layernorm

# Now import VisionMamba before both_models tries to
print("Pre-importing VisionMamba from Vim...")
try:
    from models_mamba import VisionMamba
    print(f"VisionMamba imported successfully: {VisionMamba}")
except Exception as e:
    print(f"Failed to import VisionMamba: {e}")
    raise

# Import both_models module - its try-except may leave VisionMamba undefined
# Then inject VisionMamba into its namespace before we use AMBAModel
print("Importing both_models and injecting VisionMamba...")
import models.both_models
models.both_models.VisionMamba = VisionMamba
print("VisionMamba injected into both_models namespace")


def create_test_spectrogram(seed: int = 42) -> torch.Tensor:
    """Create a deterministic test spectrogram.

    Returns shape (batch, time_frames, freq_bins) = (1, 1024, 128)
    """
    rng = np.random.default_rng(seed)
    # Simulate mel spectrogram characteristics
    spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    return torch.from_numpy(spec)


def load_original_model(checkpoint_path: str, device: str = "cuda"):
    """Load original SSAMBA model from checkpoint."""
    # Use the already-imported module with VisionMamba injected
    AMBAModel = models.both_models.AMBAModel

    # Load checkpoint to inspect config
    sd = torch.load(checkpoint_path, map_location=device)

    # Get pretraining config from checkpoint
    p_fshape = sd["module.v.patch_embed.proj.weight"].shape[2]
    p_tshape = sd["module.v.patch_embed.proj.weight"].shape[3]
    p_input_fdim = sd["module.p_input_fdim"].item()
    p_input_tdim = sd["module.p_input_tdim"].item()

    # Determine model size from embed_dim
    embed_dim = sd["module.v.pos_embed"].shape[2]
    if embed_dim == 192:
        model_size = "tiny"
    elif embed_dim == 384:
        model_size = "small"
    elif embed_dim == 768:
        model_size = "base"
    else:
        raise ValueError(f"Unknown embed_dim: {embed_dim}")

    print(f"Checkpoint config: fshape={p_fshape}, tshape={p_tshape}, "
          f"input_fdim={p_input_fdim}, input_tdim={p_input_tdim}, "
          f"embed_dim={embed_dim}, model_size={model_size}")

    # Vision Mamba config for SSAMBA
    # The pretraining script (run_mask_patch_amba.sh) uses:
    #   rms_norm='false', residual_in_fp32='false', use_middle_cls_token='false'
    vision_mamba_config = {
        "img_size": (p_input_fdim, p_input_tdim),
        "patch_size": p_fshape,  # Assuming square patches
        "stride": p_fshape // 2,
        "embed_dim": embed_dim,
        "depth": 24,
        "rms_norm": False,
        "residual_in_fp32": False,
        "fused_add_norm": False,
        "final_pool_type": "mean",
        "if_abs_pos_embed": True,
        "if_rope": False,
        "if_rope_residual": False,
        "bimamba_type": "v2",
        "if_cls_token": True,
        "if_devide_out": True,
        "use_middle_cls_token": False,
    }

    # Create model
    model = AMBAModel(
        fstride=p_fshape,
        tstride=p_tshape,
        fshape=p_fshape,
        tshape=p_tshape,
        input_fdim=p_input_fdim,
        input_tdim=p_input_tdim,
        pretrain_stage=True,
        model_size=model_size,
        vision_mamba_config=vision_mamba_config,
    )

    # Wrap in DataParallel to match checkpoint format
    model = torch.nn.DataParallel(model)
    model.load_state_dict(sd, strict=False)
    model = model.module  # Unwrap

    return model.to(device).eval()


def extract_embeddings(model, spectrogram: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Extract embeddings from the model.

    Input spectrogram: (batch, time_frames, freq_bins)
    Output: (batch, embed_dim)

    NOTE: This uses the CORRECT forward path for SSAMBA:
    - if_bidirectional=False (default) means single layer loop
    - Each layer with bimamba_type='v2' internally handles bidirectionality
    - CLS token is prepended (use_middle_cls_token=False in pretraining script)
    """
    model.eval()
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        # SSAMBA expects (batch, time_frames, freq_bins)
        # The forward method handles reshaping internally
        x = spectrogram.unsqueeze(1)  # (B, 1, T, F)
        x = x.transpose(2, 3)  # (B, 1, F, T)

        B = x.shape[0]

        # Patch embedding
        x = model.v.patch_embed(x)

        # Add CLS token at the BEGINNING (use_middle_cls_token=False)
        cls_tokens = model.v.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + model.v.pos_embed
        x = model.v.pos_drop(x)

        # CORRECT: Single layer loop (if_bidirectional=False)
        # Each layer with bimamba_type='v2' internally handles bidirectionality
        residual = None
        hidden_states = x

        print(f"Model if_bidirectional: {model.v.if_bidirectional}")
        print(f"Processing {len(model.v.layers)} layers individually (not paired)")

        for layer in model.v.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # Final norm (non-fused path for reproducibility)
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + model.v.drop_path(hidden_states)
        hidden_states = model.v.norm_f(residual.to(dtype=model.v.norm_f.weight.dtype))

        # Average all tokens except CLS (which is at position 0)
        embeddings = hidden_states[:, 1:, :].mean(dim=1)

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings with original SSAMBA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SSAMBA checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = load_original_model(args.checkpoint, args.device)

    print("Creating test spectrogram")
    spectrogram = create_test_spectrogram()

    print("Extracting embeddings")
    embeddings = extract_embeddings(model, spectrogram, args.device)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saving to {args.output}")

    # Save embeddings and metadata
    torch.save({
        "embeddings": embeddings.cpu(),
        "spectrogram": spectrogram,
        "checkpoint": args.checkpoint,
    }, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
