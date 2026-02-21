"""Model architectures for BirdSet.

Used by: Experiment 002 (Supervised ViT Baseline), Experiment 003 (Scratch Baselines)
"""

# Import metric from classifiers.py for convenience
from birdclef.classifiers import MultiLabelTopKAccuracy
from birdclef.models.audio_classifier import AudioClassifier
from birdclef.models.vit_audio import ViTAudio

__all__ = ["AudioClassifier", "ViTAudio", "MultiLabelTopKAccuracy"]
