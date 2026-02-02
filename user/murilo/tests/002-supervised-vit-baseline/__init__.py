"""Tests for Experiment 002: Supervised ViT Baseline.

Tests cover:
- Preprocessing: MelSpectrogramTransform with Kaldi fbank
- Models: ViTAudio with 1-channel patch embedding
- DataModule: BirdSetDataModule with on-the-fly mel-spectrogram generation
- Lightning: SupervisedViTModule with multi-label metrics
- Training: End-to-end training pipeline validation
"""
