"""Vision Transformer adapted for single-channel audio spectrograms.

Modifies timm's ViT-Base to accept 1-channel mel-spectrograms instead of 3-channel RGB images.
"""

import torch
import torch.nn as nn
import timm


class ViTAudio(nn.Module):
    """ViT-Base with 1-channel patch embedding for audio spectrograms.

    Args:
        num_classes: Number of output classes for classification
        img_size: Input image size as (height, width), default (128, 500) for mel-spectrograms
        patch_size: Patch size for ViT, default 16
        pretrained: Whether to initialize from ImageNet pretrained weights (default: True)
        drop_rate: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        num_classes: int,
        img_size: tuple[int, int] = (128, 500),
        patch_size: int = 16,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # Load pretrained ViT-Base from timm
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            img_size=img_size,
            drop_rate=drop_rate,
        )

        # CRITICAL: Replace 3-channel conv with 1-channel
        # Store old conv for weight initialization
        old_conv = self.backbone.patch_embed.proj

        self.backbone.patch_embed.proj = nn.Conv2d(
            in_channels=1,  # Changed from 3 (RGB) to 1 (grayscale)
            out_channels=768,  # ViT-Base embedding dimension
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Initialize from pretrained weights (average RGB channels)
        if pretrained:
            with torch.no_grad():
                # Average across RGB channels: (768, 3, 16, 16) -> (768, 1, 16, 16)
                self.backbone.patch_embed.proj.weight[:] = (
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
                if old_conv.bias is not None:
                    self.backbone.patch_embed.proj.bias[:] = old_conv.bias

        # Classification head
        embed_dim = self.backbone.embed_dim  # 768 for ViT-Base
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, 1, height, width)

        Returns:
            Logits, shape (batch, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch, embed_dim)

        # Classification
        logits = self.head(features)  # (batch, num_classes)

        return logits

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1):
        """Get intermediate layer outputs for feature extraction.

        Args:
            x: Input tensor, shape (batch, 1, height, width)
            n: Number of last blocks to return (default: 1)

        Returns:
            List of intermediate outputs from the last n transformer blocks
        """
        return self.backbone.get_intermediate_layers(x, n=n)
