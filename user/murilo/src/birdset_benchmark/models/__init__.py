"""Model architectures for BirdSet.

Used by: Experiment 002 (Supervised ViT Baseline), Experiment 003 (Scratch Baselines)
"""

# Import metric from classifiers.py for convenience
from birdset_benchmark.classifiers import MultiLabelTopKAccuracy
from birdset_benchmark.models.audio_classifier import AudioClassifier
from birdset_benchmark.models.vit_audio import ViTAudio

__all__ = ["AudioClassifier", "ViTAudio", "MultiLabelTopKAccuracy"]
