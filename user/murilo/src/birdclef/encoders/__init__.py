"""Pretrained audio encoder wrappers."""

from .convnext import ConvNeXtEncoder
from .eat import EATEncoder
from .efficientnet import EfficientNetEncoder

__all__ = ["ConvNeXtEncoder", "EATEncoder", "EfficientNetEncoder"]

# BEATsEncoder requires vendor setup — import explicitly when needed:
#   from birdclef.encoders.beats import BEATsEncoder
