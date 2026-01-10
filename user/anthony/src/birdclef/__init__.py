"""BirdCLEF encoder benchmarking and audio classification."""

from .audio import load_audio, pad_or_trim
from .data import ESC50Dataset
from .encoders import AudioEncoder, Wav2Vec2Encoder, BEATsEncoder, EATEncoder
from .train import LinearProbe, evaluate_5fold_cv

__all__ = [
    "load_audio",
    "pad_or_trim",
    "ESC50Dataset",
    "AudioEncoder",
    "Wav2Vec2Encoder",
    "BEATsEncoder",
    "EATEncoder",
    "LinearProbe",
    "evaluate_5fold_cv",
]
