"""EfficientNet encoder wrapper for 1-channel mel-spectrogram input.

Uses timm's pretrained EfficientNet models with features_only=True to
return per-stage feature maps, enabling SED (Sound Event Detection)
architectures that need spatial/temporal feature maps instead of pooled
embeddings.
"""

import logging

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EfficientNetEncoder(nn.Module):
    """Timm EfficientNet backbone adapted for 1-channel mel-spectrogram input.

    Creates an EfficientNet model with features_only=True (returns per-stage
    feature maps) and adapts the stem conv from 3 channels to 1 channel.

    The forward_features() method returns the last-stage 4D feature map for
    SED heads. The forward() method returns a pooled embedding for backward
    compatibility with clip-level classification.

    Args:
        model_name: Timm model name (e.g., "efficientnet_b0", "efficientnet_b3")
        pretrained: Whether to load pretrained ImageNet weights
        drop_path_rate: Stochastic depth rate (default: 0.0 = disabled).
            Randomly drops entire residual blocks during training.
            2025 1st place used 0.15, Tom used 0.0 (default).
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, drop_path_rate=drop_path_rate
        )
        self.embed_dim = self.backbone.feature_info.channels()[-1]  # 1280 for B0
        self._adapt_input_channels(pretrained)

        dp_info = f", drop_path={drop_path_rate}" if drop_path_rate > 0 else ""
        logger.info(
            f"EfficientNetEncoder: {model_name}, embed_dim={self.embed_dim}, "
            f"stages={len(self.backbone.feature_info.channels())}{dp_info}"
        )

    def _adapt_input_channels(self, pretrained: bool):
        """Replace 3-channel stem conv with 1-channel variant.

        EfficientNet's stem conv is at self.backbone.conv_stem. If using
        pretrained weights, average the 3-channel weights into 1 channel
        to preserve learned features.
        """
        old_conv = self.backbone.conv_stem
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
        self.backbone.conv_stem = new_conv

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last-stage feature map for SED heads.

        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)

        Returns:
            Feature map, shape (batch, embed_dim, freq', time') where freq' and
            time' are the spatially downsampled dimensions. For EfficientNet-B0
            with input (B, 1, 128, 1000): output ~(B, 1280, 4, 31).
        """
        feature_maps = self.backbone(x)  # list of per-stage outputs
        return feature_maps[-1]  # last stage: highest-level features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible pooled embedding.

        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)

        Returns:
            Feature vector, shape (batch, embed_dim)
        """
        features = self.forward_features(x)
        return features.mean(dim=(-2, -1))  # global average pool
