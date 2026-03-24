"""Sound Event Detection (SED) head with attention-based temporal pooling.

Architecture based on the BirdCLEF 2025 1st place solution:
  backbone features (B, C, F, T)
    → GeM pool frequency axis → (B, C, T)
    → dense projection + dropout → (B, hidden, T)
    → framewise logits via Conv1d → (B, num_classes, T)
    → attention weights via Conv1d + softmax → (B, num_classes, T)
    → clipwise = attention-weighted sum → (B, num_classes)

Training uses dual supervision on clipwise and framewise-max predictions.
Inference blends 75% clipwise + 25% framewise-max.
"""

import torch
import torch.nn as nn


class GeM(nn.Module):
    """Generalized Mean Pooling on the frequency dimension.

    Collapses the second-to-last dimension (frequency) with a learnable power
    parameter p. At p=1 this is average pooling; as p→∞ it approaches max pooling.

    Args:
        p: Initial power parameter (default: 3.0).
        eps: Small constant for numerical stability.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool frequency dimension: (B, C, F, T) → (B, C, T)."""
        x_clamped = x.clamp(min=self.eps)
        return x_clamped.pow(self.p).mean(dim=-2).pow(1.0 / self.p)


class AttHead(nn.Module):
    """SED attention head with GeM frequency pooling and dual output.

    Produces two sets of logits for dual supervision:
    - clipwise: attention-weighted temporal aggregation (captures sustained calls)
    - framewise: per-frame predictions (framewise-max captures brief vocalizations)

    Args:
        in_channels: Number of input channels from backbone (e.g., 1280 for EfficientNet-B0).
        num_classes: Number of output classes (e.g., 234).
        hidden_dim: Hidden projection dimension (default: 512).
        dropout: Dropout rate for projection layer (default: 0.5).
        gem_p: Initial GeM power parameter (default: 3.0).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.gem = GeM(p=gem_p)
        self.project = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.frame_logits_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.attention_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from backbone features to dual logits.

        Args:
            features: (B, C, F, T) feature map from backbone.

        Returns:
            clipwise_logits: (B, num_classes) attention-weighted temporal sum.
            framewise_logits: (B, num_classes, T) per-frame predictions.
        """
        # GeM pool frequency: (B, C, F, T) → (B, C, T)
        x = self.gem(features)

        # Dense projection: (B, C, T) → permute → Linear → permute → (B, hidden, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.project(x)  # (B, T, hidden)
        x = x.permute(0, 2, 1)  # (B, hidden, T)

        # Per-frame logits and attention weights
        frame_logits = self.frame_logits_conv(x)  # (B, num_classes, T)
        att_weights = torch.softmax(self.attention_conv(x), dim=-1)  # (B, num_classes, T)

        # Clipwise: attention-weighted sum of frame logits over time
        clipwise_logits = (att_weights * frame_logits).sum(dim=-1)  # (B, num_classes)

        return clipwise_logits, frame_logits
