"""Pretrained audio encoder wrappers."""

from .eat import EATEncoder

__all__ = ["EATEncoder"]

# BEATsEncoder requires vendor setup — import explicitly when needed:
#   from birdclef.encoders.beats import BEATsEncoder
