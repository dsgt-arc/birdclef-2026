---
slug: "shionao7/bird-25-submission-regnety008-v1"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 269
---

# Kaggle Summary: Bird'25 Submission RegNetY008 v1

**Source**: https://www.kaggle.com/code/shionao7/bird-25-submission-regnety008-v1
**Competition**: BirdCLEF 2025
**Score**: Unknown (inference-only notebook)

## Summary

This notebook provides a CPU-based inference pipeline for BirdCLEF 2025 using a RegNetY-008 architecture pre-trained on the competition data. The approach processes 5-second audio windows from soundscape recordings, converts them to mel spectrograms, and uses an ensemble of trained models with temporal smoothing to generate final predictions.

## Approach

### Architecture

The model uses a RegNetY-008 backbone from the timm library configured for single-channel input. The architecture replaces the standard classifier head with a custom design that includes adaptive average pooling followed by a linear layer outputting logits for all bird species classes. The backbone is initialized without ImageNet pre-training, indicating it was trained from scratch on bird audio spectrograms.

### Data Pipeline

Audio processing uses librosa to load files at 32kHz sampling rate, then converts 5-second windows into mel spectrograms with the following parameters:
- FFT size: 1024
- Hop length: 256
- Mel bins: 256
- Frequency range: 50-14000 Hz
- Target shape: 256x256 pixels

The mel spectrograms are normalized to the 0-1 range using min-max scaling after converting to decibel scale. Missing or NaN values in audio are replaced with the mean signal value. Spectrograms are resized to the exact target shape using OpenCV linear interpolation if needed.

### Training

Training details are not included in this inference-only notebook. The pipeline loads pre-trained checkpoint files containing model state dictionaries, suggesting K-fold cross-validation was used during training based on fold-specific file naming conventions. The notebook supports optional fold selection to use specific trained models from the ensemble.

### Inference / Post-Processing

The inference pipeline includes several key features:

**Model Ensemble**: Multiple fold-specific models can be loaded and their predictions averaged. The notebook supports either using all available models or selecting specific folds.

**Test Time Augmentation**: Optional TTA applies three variations to each spectrogram - no augmentation, horizontal flip (time reversal), and vertical flip (frequency reversal). Predictions from all variations are averaged.

**Temporal Smoothing**: A post-processing step applies weighted averaging across adjacent time segments within each soundscape. The smoothing uses different weights for boundary segments (80% current + 20% neighbor) versus interior segments (20% previous + 60% current + 20% next). This enforces temporal consistency in predictions.

The pipeline processes soundscapes segment-by-segment, generating predictions for each 5-second window and applying sigmoid activation to convert logits to probabilities.

## Key Ideas

- **Frequency band selection**: Uses 50-14000 Hz range instead of full spectrum, potentially focusing on the most informative frequency bands for bird vocalizations
- **Temporal smoothing**: Post-hoc averaging of predictions across adjacent time windows helps reduce prediction noise and enforce temporal coherence
- **CPU-optimized inference**: Designed to run entirely on CPU without GPU requirements, using efficient model architecture and processing
- **Flexible ensemble**: Architecture supports both single-model and multi-model ensemble inference with configurable fold selection
- **Three-way TTA**: Includes both time and frequency flip augmentations in addition to the original spectrogram

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | Not reported |
| Votes  | 269 |

## Connections

- **Related concepts**: Relevant to temporal aggregation strategies and mel spectrogram parameter choices
- **Informs experiments**: The temporal smoothing approach could be evaluated against other aggregation methods; frequency range selection (50-14k Hz) differs from our configurations
- **Similar approaches**: This represents a standard timm-based approach similar to common competition baselines but with focused parameter tuning
