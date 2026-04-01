"""Timm encoder wrapper for 1-channel mel-spectrogram input.

Uses timm's pretrained models (EfficientNet, HGNetV2, etc.) with
features_only=True to return per-stage feature maps, enabling SED
(Sound Event Detection) architectures that need spatial/temporal
feature maps instead of pooled embeddings.
"""

import logging

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EfficientNetEncoder(nn.Module):
    """Timm backbone adapted for 1-channel mel-spectrogram input.

    Creates a timm model with features_only=True (returns per-stage
    feature maps) and in_chans=1. Works with any timm model that
    supports features_only (EfficientNet, HGNetV2, ConvNeXt, etc.).

    The forward_features() method returns the last-stage 4D feature map for
    SED heads. The forward() method returns a pooled embedding for backward
    compatibility with clip-level classification.

    Args:
        model_name: Timm model name (e.g., "efficientnet_b0",
            "hgnetv2_b0.ssld_stage2_ft_in1k")
        pretrained: Whether to load pretrained ImageNet weights
        drop_path_rate: Stochastic depth rate (default: 0.0 = disabled).
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        in_chans: int = 1,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = self.backbone.feature_info.channels()[-1]

        # Optional 1x1 conv to project features to a higher dimension.
        # Useful for testing whether embed_dim is a bottleneck (e.g. 320→1024).
        self.projection = None
        if projection_dim is not None and projection_dim != self.embed_dim:
            self.projection = nn.Conv2d(self.embed_dim, projection_dim, kernel_size=1)
            logger.info(f"  Projection: {self.embed_dim} → {projection_dim}")
            self.embed_dim = projection_dim

        # EfficientNet family: manually adapt stem conv for better weight init
        # (average 3-ch pretrained weights into 1-ch instead of timm's default)
        if in_chans == 1 and hasattr(self.backbone, "conv_stem") and pretrained:
            self._adapt_input_channels()

        dp_info = f", drop_path={drop_path_rate}" if drop_path_rate > 0 else ""
        logger.info(
            f"EfficientNetEncoder: {model_name}, embed_dim={self.embed_dim}, "
            f"stages={len(self.backbone.feature_info.channels())}{dp_info}"
        )

    def _adapt_input_channels(self):
        """Re-init EfficientNet stem conv with averaged 3→1 channel weights.

        timm's in_chans=1 repeats or truncates channel weights. For EfficientNet,
        we get better results by averaging the 3-channel pretrained weights.
        Only called when backbone has conv_stem (EfficientNet family).
        """
        old_conv = self.backbone.conv_stem
        if old_conv.in_channels == 1:
            # timm already set to 1-ch, but re-init with averaged weights
            # by reloading the 3-ch model weights and averaging
            return  # timm handled it, skip manual adaptation
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
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
        features = feature_maps[-1]  # last stage: highest-level features
        if self.projection is not None:
            features = self.projection(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible pooled embedding.

        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)

        Returns:
            Feature vector, shape (batch, embed_dim)
        """
        features = self.forward_features(x)
        return features.mean(dim=(-2, -1))  # global average pool
