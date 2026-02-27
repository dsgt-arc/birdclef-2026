"""ConvNeXt encoder wrapper for 1-channel mel-spectrogram input.

Uses timm's pretrained ConvNeXt models as feature extractors,
adapting the 3-channel ImageNet stem to 1-channel audio input.
"""

import timm
import torch
import torch.nn as nn


class ConvNeXtEncoder(nn.Module):
    """Timm ConvNeXt backbone adapted for 1-channel mel-spectrogram input.

    Creates a ConvNeXt model from timm with num_classes=0 (feature extractor only),
    then adapts the stem conv from 3 channels to 1 channel by averaging pretrained
    weights across the input channel dimension.

    Args:
        model_name: Timm model name (e.g., "convnext_base.fb_in22k_ft_in1k")
        pretrained: Whether to load pretrained ImageNet weights
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features  # 1024 for ConvNeXt-Base
        self._adapt_input_channels(pretrained)

    def _adapt_input_channels(self, pretrained: bool):
        """Replace 3-channel stem conv with 1-channel variant.

        ConvNeXt stem[0] is the input Conv2d. If using pretrained weights,
        average the 3-channel weights into 1 channel to preserve learned features.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: mel spectrogram -> feature vector.

        Args:
            x: Input tensor, shape (batch, 1, 128, 500)

        Returns:
            Feature vector, shape (batch, embed_dim)
        """
        return self.backbone(x)
