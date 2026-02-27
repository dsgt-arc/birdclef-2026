"""Pretrained audio encoder wrappers."""

from .convnext import ConvNeXtEncoder
from .eat import EATEncoder

__all__ = ["ConvNeXtEncoder", "EATEncoder"]

# BEATsEncoder requires vendor setup — import explicitly when needed:
#   from birdclef.encoders.beats import BEATsEncoder
