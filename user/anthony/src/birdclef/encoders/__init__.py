"""Audio encoder wrappers for feature extraction."""

from .base import AudioEncoder
from .wav2vec2 import Wav2Vec2Encoder
from .beats import BEATsEncoder
from .eat import EATEncoder
from .ast import ASTEncoder

# Conditional imports for models with complex dependencies
try:
    from .ssast import SSASTEncoder
except ImportError:
    SSASTEncoder = None

try:
    from .ssamba import SSAMBAEncoder
except ImportError:
    SSAMBAEncoder = None

__all__ = [
    "AudioEncoder",
    "Wav2Vec2Encoder",
    "BEATsEncoder",
    "EATEncoder",
    "ASTEncoder",
    "SSASTEncoder",
    "SSAMBAEncoder",
]
