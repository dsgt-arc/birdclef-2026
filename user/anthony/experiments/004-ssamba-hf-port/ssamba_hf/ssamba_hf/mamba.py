"""Mamba layers for SSAMBA model.

This module implements bidirectional Mamba blocks using HuggingFace's MambaMixer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer

from .config import SSAMBAConfig
from .layers import RMSNorm, DropPath


def create_mamba_config(ssamba_config: SSAMBAConfig) -> MambaConfig:
    """Create HuggingFace MambaConfig from SSAMBAConfig."""
    return MambaConfig(
        hidden_size=ssamba_config.embed_dim,
        state_size=ssamba_config.d_state,
        num_hidden_layers=1,  # We manage layers ourselves
        expand=ssamba_config.expand,
        conv_kernel=ssamba_config.d_conv,
        use_bias=True,
        use_conv_bias=True,
        hidden_act="silu",
        residual_in_fp32=ssamba_config.residual_in_fp32,
        use_cache=False,
    )


class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm.

    This matches VisionMamba's Block structure:
    - Pre-norm with RMSNorm
    - Mamba mixer
    - Residual connection
    """

    def __init__(
        self,
        config: SSAMBAConfig,
        mamba_config: MambaConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = RMSNorm(config.embed_dim)
        self.mixer = MambaMixer(mamba_config, layer_idx=layer_idx)
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
        # Pre-norm pattern matching VisionMamba
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            # Fused add + norm
            if self.residual_in_fp32:
                residual = residual.float()
            residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(hidden_states.dtype))

        # Mamba mixer
        hidden_states = self.mixer(hidden_states)

        return hidden_states, residual


class BidirectionalMambaStack(nn.Module):
    """Stack of bidirectional Mamba blocks.

    Implements VisionMamba's bimamba_type="v2" pattern:
    - Layers are paired (even=forward, odd=backward)
    - Forward and backward outputs are summed
    """

    def __init__(self, config: SSAMBAConfig):
        super().__init__()
        self.config = config

        mamba_config = create_mamba_config(config)

        # Create paired layers (forward, backward)
        self.layers = nn.ModuleList([
            MambaBlock(config, mamba_config, layer_idx=i)
            for i in range(config.depth)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through bidirectional Mamba stack.

        Args:
            hidden_states: Input tensor (batch, seq_len, embed_dim)

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        residual = None

        # Process in pairs: even layers forward, odd layers backward
        for i in range(len(self.layers) // 2):
            forward_layer = self.layers[i * 2]
            backward_layer = self.layers[i * 2 + 1]

            # Forward direction
            hidden_states_f, residual_f = forward_layer(hidden_states, residual)

            # Backward direction (flip sequence, process, flip back)
            hidden_states_b, residual_b = backward_layer(
                hidden_states.flip([1]),
                None if residual is None else residual.flip([1])
            )

            # Combine forward and backward
            hidden_states = hidden_states_f + hidden_states_b.flip([1])
            residual = residual_f + residual_b.flip([1])

        # Final residual connection
        if residual is not None:
            hidden_states = residual + hidden_states

        return hidden_states
