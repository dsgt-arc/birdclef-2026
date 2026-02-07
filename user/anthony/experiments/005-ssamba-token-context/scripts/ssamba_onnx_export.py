"""ONNX export utilities for SSAMBA with mamba-onnx custom ops.

Replaces the pure-PyTorch selective scan loop with a single
mamba::SelectiveScanFusedExact custom op node during ONNX export.
The custom op is provided by the mamba-onnx package and runs optimized
AVX2+OpenMP kernels at inference time via ONNX Runtime.

Tensor layout expected by mamba-onnx (from mamba_onnx tests/test_ops.py):
  u:     [B, d_inner, L]
  delta: [B, d_inner, L]
  A:     [d_inner, d_state]
  B:     [B, L, d_state]
  C:     [B, L, d_state]
  D:     [d_inner]
  z:     [B, d_inner, L]
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add ssamba_hf to path
ssamba_hf_dir = Path(__file__).resolve().parents[2] / "004-ssamba-hf-port" / "ssamba_hf"
sys.path.insert(0, str(ssamba_hf_dir))

from ssamba_hf.mamba import BidirectionalMamba


class SelectiveScanONNXExport(torch.autograd.Function):
    """Emit a single mamba::SelectiveScanFusedExact node during ONNX export.

    forward() returns a shape stub (zeros) — actual compute is done by
    ONNX Runtime's custom op at inference time.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z):
        # Shape stub: output has same shape as u [B, d_inner, L]
        return torch.zeros_like(u)

    @staticmethod
    def symbolic(g, u, delta, A, B, C, D, z):
        return g.op(
            "mamba::SelectiveScanFusedExact",
            u, delta, A, B, C, D, z,
        )


def _ssm_forward_onnx(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
    conv1d: nn.Conv1d,
    x_proj: nn.Linear,
    dt_proj: nn.Linear,
    A_log: nn.Parameter,
    D: nn.Parameter,
) -> torch.Tensor:
    """ONNX-exportable replacement for BidirectionalMamba._ssm_forward_pytorch.

    Standard ops (conv1d, silu, linear projections) trace normally.
    The selective scan recurrence becomes a single custom op node.

    Args/Returns match _ssm_forward_pytorch exactly.
    """
    B, L, d_inner = x.shape

    # Conv1d: (B, L, D) -> (B, D, L) -> conv -> truncate -> silu
    x = x.transpose(1, 2)
    x = conv1d(x)[:, :, :L]
    x = F.silu(x)

    # Project to dt, B_mat, C_mat via standard linear ops
    x_dbl = x_proj(x.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
    dt, B_mat, C_mat = x_dbl.split(
        [self.dt_rank, self.d_state, self.d_state], dim=-1
    )

    # dt_proj WITHOUT softplus — the fused op handles softplus internally.
    # Include bias here since selective_scan_fn expects delta_bias separately,
    # but the fused ONNX op expects it baked into delta.
    dt = F.linear(dt, dt_proj.weight, dt_proj.bias)  # (B, L, d_inner)
    dt = dt.transpose(1, 2)  # (B, d_inner, L)

    # A from log-parameterization
    A = -torch.exp(A_log.float())  # (d_inner, d_state)

    # Prepare z: (B, L, d_inner) -> (B, d_inner, L)
    z = z.transpose(1, 2).contiguous()

    # u is already (B, d_inner, L) from the conv1d path
    u = x.contiguous()

    # B_mat: (B, L, d_state) — already in correct layout
    # C_mat: (B, L, d_state) — already in correct layout

    # Single custom op node replaces the entire sequential scan + gating
    y = SelectiveScanONNXExport.apply(u, dt, A, B_mat, C_mat, D.float(), z)

    # Output: (B, d_inner, L) -> (B, L, d_inner)
    return y.transpose(1, 2)


def export_ssamba(model, dummy_input, output_path, opset_version=17):
    """Export an SSAMBAModel to ONNX with mamba-onnx custom ops.

    Monkey-patches BidirectionalMamba._ssm_forward_pytorch with the ONNX-
    exportable version during export, then restores the original.

    Args:
        model: SSAMBAModel instance (should be on CPU, eval mode)
        dummy_input: Example input tensor (B, time_frames, mel_bins)
        output_path: Where to save the .onnx file
        opset_version: ONNX opset version (default 17)
    """
    original_method = BidirectionalMamba._ssm_forward_pytorch

    try:
        BidirectionalMamba._ssm_forward_pytorch = _ssm_forward_onnx

        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            opset_version=opset_version,
            custom_opsets={"mamba": 1},
            input_names=["input"],
            output_names=["embeddings"],
            dynamic_axes={
                "input": {0: "batch"},
                "embeddings": {0: "batch"},
            },
            dynamo=False,
        )
    finally:
        BidirectionalMamba._ssm_forward_pytorch = original_method
