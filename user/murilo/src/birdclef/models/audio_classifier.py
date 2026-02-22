"""Generic audio classifier supporting ViT and ConvNeXT families.

Used by: Experiment 003 (Scratch Baselines)

Adapts timm models to accept 1-channel mel-spectrograms instead of 3-channel RGB images.
Supports both ViT (patch_embed.proj) and ConvNeXT (stem[0]) input layer adaptation.
"""

import timm
import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    """Generic audio classifier with 1-channel input adaptation.

    Supports ViT and ConvNeXT model families from timm.

    Args:
        model_name: timm model name (e.g., 'vit_small_patch16_224', 'convnext_tiny')
        num_classes: Number of output classes for classification
        img_size: Input image size as (height, width), default (128, 500) for mel-spectrograms
        pretrained: Whether to initialize from pretrained weights (default: False)
        drop_rate: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        img_size: tuple[int, int] = (128, 500),
        pretrained: bool = False,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # Create backbone from timm
        create_kwargs = {"pretrained": pretrained, "num_classes": 0, "drop_rate": drop_rate}
        if "vit" in model_name:
            create_kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **create_kwargs)

        # Adapt input from 3 channels to 1 channel
        self._adapt_input_channels(pretrained)

        # Detect embedding dimension
        if hasattr(self.backbone, "embed_dim"):
            embed_dim = self.backbone.embed_dim  # ViT family
        elif hasattr(self.backbone, "num_features"):
            embed_dim = self.backbone.num_features  # ConvNeXT family
        else:
            raise ValueError(f"Cannot detect embed_dim for model: {model_name}")

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _adapt_input_channels(self, pretrained: bool):
        """Replace 3-channel input conv with 1-channel variant."""
        if hasattr(self.backbone, "patch_embed"):
            # ViT family: patch_embed.proj is the input Conv2d
            old_conv = self.backbone.patch_embed.proj
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                bias=old_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        new_conv.bias[:] = old_conv.bias
            self.backbone.patch_embed.proj = new_conv

        elif hasattr(self.backbone, "stem"):
            # ConvNeXT family: stem[0] is the input Conv2d
            old_conv = self.backbone.stem[0]
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        new_conv.bias[:] = old_conv.bias
            self.backbone.stem[0] = new_conv

        else:
            raise ValueError("Unsupported architecture: no patch_embed or stem found")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, 1, height, width)

        Returns:
            Logits, shape (batch, num_classes)
        """
        features = self.backbone(x)  # (batch, embed_dim)
        logits = self.head(features)  # (batch, num_classes)
        return logits
