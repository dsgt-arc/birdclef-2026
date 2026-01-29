"""Mamba layers for SSAMBA model.

This module implements bidirectional Mamba blocks matching VisionMamba's bimamba_type="v2".
Uses pure PyTorch by default, with optional mamba-ssm CUDA kernel support for ~100x speedup.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import SSAMBAConfig
from .layers import RMSNorm, DropPath

# CUDA kernel availability detection
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    selective_scan_fn = None
    MAMBA_SSM_AVAILABLE = False


def is_cuda_kernel_available() -> bool:
    """Check if mamba-ssm CUDA kernels are available."""
    return MAMBA_SSM_AVAILABLE


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba (bimamba_type=v2) with optional CUDA kernel support.

    This matches the VisionMamba's Mamba layer structure with both forward
    and backward processing weights within a single module.

    Weight structure:
        Forward: A_log, D, conv1d, x_proj, dt_proj
        Backward: A_b_log, D_b, conv1d_b, x_proj_b, dt_proj_b
        Shared: in_proj, out_proj

    Args:
        use_cuda_kernels: Optional[bool]
            - None (default): Auto-detect, use CUDA kernels if available and on GPU
            - True: Require CUDA kernels, raise error if unavailable
            - False: Force pure PyTorch implementation
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
        use_cuda_kernels: Optional[bool] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._use_cuda_kernels = use_cuda_kernels
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # Input projection (shared)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Forward direction weights
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # S4D-style initialization for A and D
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        # Backward direction weights (suffix _b)
        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Backward A and D
        A_b = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(self.d_inner, -1)
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))

        # Output projection (shared)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _should_use_cuda_kernels(self, x: torch.Tensor) -> bool:
        """Determine whether to use CUDA kernels for this forward pass."""
        if self._use_cuda_kernels is False:
            return False
        if self._use_cuda_kernels is True:
            if not MAMBA_SSM_AVAILABLE:
                raise RuntimeError("use_cuda_kernels=True but mamba-ssm not installed")
            if not x.is_cuda:
                raise RuntimeError("use_cuda_kernels=True but tensor not on CUDA device")
            return True
        # Auto mode: use CUDA if available and tensor is on GPU
        return MAMBA_SSM_AVAILABLE and x.is_cuda

    def _ssm_forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A_log: nn.Parameter,
        D: nn.Parameter,
    ) -> torch.Tensor:
        """Selective scan dispatcher - routes to CUDA or PyTorch backend."""
        if self._should_use_cuda_kernels(x):
            return self._ssm_forward_cuda(x, z, conv1d, x_proj, dt_proj, A_log, D)
        return self._ssm_forward_pytorch(x, z, conv1d, x_proj, dt_proj, A_log, D)

    def _ssm_forward_cuda(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A_log: nn.Parameter,
        D: nn.Parameter,
    ) -> torch.Tensor:
        """CUDA-accelerated selective scan using mamba-ssm kernels.

        Args:
            x: Input after in_proj, shape (B, L, d_inner)
            z: Gating signal, shape (B, L, d_inner)
            conv1d: 1D convolution layer
            x_proj: Projection to dt, B, C
            dt_proj: Delta projection
            A_log: Log of A matrix
            D: Skip connection

        Returns:
            Output tensor, shape (B, L, d_inner)
        """
        B, L, d_inner = x.shape

        # Conv1d (transpose for (B, D, L) layout)
        x = x.transpose(1, 2)
        x = conv1d(x)[:, :, :L]
        x = F.silu(x)

        # Project to get delta, B, C
        x_dbl = x_proj(x.transpose(1, 2))
        dt, B_mat, C_mat = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.transpose(1, 2)  # (B, d_inner, L)

        # Prepare for selective_scan_fn (expects specific tensor layouts)
        A = -torch.exp(A_log.float())  # (d_inner, d_state)
        B_mat = B_mat.transpose(1, 2).contiguous()  # (B, d_state, L)
        C_mat = C_mat.transpose(1, 2).contiguous()  # (B, d_state, L)
        z = z.transpose(1, 2).contiguous()  # (B, d_inner, L)

        # Call CUDA kernel
        y = selective_scan_fn(
            u=x.contiguous(),
            delta=dt.contiguous(),
            A=A.contiguous(),
            B=B_mat,
            C=C_mat,
            D=D.float(),
            z=z,
            delta_bias=dt_proj.bias.float() if dt_proj.bias is not None else None,
            delta_softplus=True,
            return_last_state=False,
        )

        return y.transpose(1, 2)

    def _ssm_forward_pytorch(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A_log: nn.Parameter,
        D: nn.Parameter,
    ) -> torch.Tensor:
        """Pure PyTorch selective scan (one direction).

        Args:
            x: Input after in_proj, shape (B, L, d_inner)
            z: Gating signal, shape (B, L, d_inner)
            conv1d: 1D convolution layer
            x_proj: Projection to dt, B, C
            dt_proj: Delta projection
            A_log: Log of A matrix
            D: Skip connection

        Returns:
            Output tensor, shape (B, L, d_inner)
        """
        B, L, d_inner = x.shape

        # 1D convolution (transpose for conv1d: B, L, D -> B, D, L)
        x = x.transpose(1, 2)
        x = conv1d(x)[:, :, :L]  # Causal: truncate to original length
        x = x.transpose(1, 2)

        # SiLU activation
        x = F.silu(x)

        # Project to get delta, B, C
        x_dbl = x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B_mat, C_mat = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Delta projection
        dt = dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive

        # Get A from log
        A = -torch.exp(A_log.float())  # (d_inner, d_state)

        # Discretize: A_bar = exp(delta * A)
        # For efficiency, we compute the selective scan directly
        # This is a simplified pure-PyTorch implementation
        # Note: This may be slower than the CUDA kernel but is functionally equivalent

        # State evolution: h_{t+1} = A_bar * h_t + B_bar * x_t
        # Output: y_t = C_t * h_t + D * x_t

        # Initialize hidden state
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            # Get time-step values
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            B_t = B_mat[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            C_t = C_mat[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)

            # Discretized A: exp(dt * A)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)

            # Discretized B: dt * B (simplified Euler)
            B_bar = dt_t * B_t  # (B, d_inner, d_state)

            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t  # (B, d_inner, d_state)

            # Output: y = (C * h).sum(-1) + D * x
            y_t = (C_t * h).sum(-1) + D * x[:, t, :]  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        # Gating with z
        y = y * F.silu(z)

        return y

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Bidirectional forward pass.

        Args:
            hidden_states: Input tensor (B, L, d_model)

        Returns:
            Output tensor (B, L, d_model)
        """
        B, L, _ = hidden_states.shape

        # Input projection
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Forward direction
        y_f = self._ssm_forward(x, z, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)

        # Backward direction (flip, process, flip back)
        x_b = x.flip([1])
        z_b = z.flip([1])
        y_b = self._ssm_forward(x_b, z_b, self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b)
        y_b = y_b.flip([1])

        # Combine forward and backward (average, matching VisionMamba's if_devide_out=True)
        y = (y_f + y_b) / 2

        # Output projection
        out = self.out_proj(y)

        return out


class MambaBlock(nn.Module):
    """Single bidirectional Mamba block with pre-norm.

    This matches VisionMamba's Block structure:
    - Pre-norm with RMSNorm or LayerNorm (based on config.rms_norm)
    - Bidirectional Mamba mixer
    - Residual connection with drop path
    """

    def __init__(
        self,
        config: SSAMBAConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.rms_norm:
            self.norm = RMSNorm(config.embed_dim)
        else:
            self.norm = nn.LayerNorm(config.embed_dim)
        self.mixer = BidirectionalMamba(
            d_model=config.embed_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            layer_idx=layer_idx,
            use_cuda_kernels=config.use_cuda_kernels,
        )
        self.drop_path = DropPath(config.drop_path_rate)
        self.residual_in_fp32 = config.residual_in_fp32

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual tracking.

        Args:
            hidden_states: Input tensor (batch, seq_len, embed_dim)
            residual: Previous residual tensor, or None

        Returns:
            Tuple of (hidden_states, residual)
        """
        # Pre-norm pattern matching VisionMamba's non-fused path
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)

        # Normalize for mixer input
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        # Convert residual to fp32 AFTER norm (matching original)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # Bidirectional Mamba mixer
        hidden_states = self.mixer(hidden_states)

        return hidden_states, residual


class BidirectionalMambaStack(nn.Module):
    """Stack of bidirectional Mamba blocks.

    With bimamba_type="v2", each block internally handles both directions.
    The depth parameter specifies the number of blocks.
    """

    def __init__(self, config: SSAMBAConfig):
        super().__init__()
        self.config = config

        # Create blocks - each block is bidirectional
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=i)
            for i in range(config.depth)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through bidirectional Mamba stack.

        Args:
            hidden_states: Input tensor (batch, seq_len, embed_dim)

        Returns:
            Tuple of (hidden_states, residual) for final norm handling
        """
        residual = None

        # Process through all layers
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # Return both for proper final norm handling
        return hidden_states, residual
