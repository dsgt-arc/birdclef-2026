"""Audio encoder wrappers for feature extraction."""

from .base import AudioEncoder
from .wav2vec2 import Wav2Vec2Encoder
from .beats import BEATsEncoder
from .eat import EATEncoder

__all__ = ["AudioEncoder", "Wav2Vec2Encoder", "BEATsEncoder", "EATEncoder"]
