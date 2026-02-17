---
slug: "kumarandatascientist/lb-0-804-efficientnet-b0-pytorch-pipeline"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 385
---

# Kaggle Summary: LB 0.804 EfficientNet B0 PyTorch Pipeline

**Source**: https://www.kaggle.com/code/kumarandatascientist/lb-0-804-efficientnet-b0-pytorch-pipeline
**Competition**: BirdCLEF 2025
**Score**: 0.804 (Public LB)

## Summary

This inference-only notebook implements an EfficientNet-B0 based pipeline for soundscape prediction with test-time augmentation and temporal smoothing. It processes 5-second audio segments from soundscape recordings, predicts species presence using pre-trained models, and applies post-processing to improve temporal consistency across consecutive segments.

## Approach

### Architecture

The model uses a standard EfficientNet-B0 backbone from the timm library with a custom classification head. The backbone is configured for single-channel input (grayscale mel spectrograms) and uses global average pooling followed by a linear classifier to predict multi-label species presence. The architecture supports both single-model and ensemble inference, averaging predictions across multiple folds.

### Data Pipeline

Audio is loaded at 32 kHz sampling rate and processed into 5-second windows. Each segment is converted to a mel spectrogram using librosa with the following parameters:
- N_FFT: 1034
- Hop length: 64
- Number of mel bins: 136
- Frequency range: 20-16000 Hz

The mel spectrograms are converted to decibel scale, min-max normalized, and resized to 256x256 using OpenCV linear interpolation. Audio segments shorter than 5 seconds are zero-padded. NaN values in audio are replaced with the mean signal value.

### Training

Training details are not included in this inference notebook. The notebook loads pre-trained model checkpoints in `.pth` format from a Kaggle dataset. It supports loading multiple model files for ensemble prediction and can optionally filter to specific cross-validation folds.

### Inference / Post-Processing

The notebook implements three key inference techniques:

1. **Test-time augmentation (TTA)**: Three augmentation variations are applied - no augmentation, horizontal flip (time shift), and vertical flip (frequency shift). Predictions are averaged across all TTA variations.

2. **Ensemble averaging**: When multiple model checkpoints are loaded, predictions from all models are averaged for each segment.

3. **Temporal smoothing**: After initial predictions, a post-processing step smooths predictions across consecutive segments within each soundscape file. For interior segments, predictions are weighted as 20% previous + 60% current + 20% next. Edge segments use 80% current + 20% neighbor weighting.

## Key Ideas

- **Optimized mel spectrogram parameters**: Uses hop length of 64 (down from typical 512-1024) and 136 mel bins to capture finer temporal detail while maintaining reasonable computational cost
- **Conservative TTA strategy**: Only uses simple flip augmentations rather than more complex transformations that might degrade signal quality
- **Temporal smoothing as post-processing**: Enforces smoothness across time by averaging predictions with neighboring segments, exploiting the assumption that species presence changes gradually in soundscape recordings
- **Modular pipeline design**: The code is organized into a comprehensive class structure that handles taxonomy loading, model management, audio processing, and prediction in a clean, extensible way

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.804 |
| Private LB | Not reported |

## Connections

- **Related concepts**: Temporal smoothing for soundscape inference, test-time augmentation strategies
- **Informs experiments**: The temporal smoothing technique and optimized mel parameters could be adapted for SSAMBA soundscape inference
- **Similar approaches**: This represents a strong EfficientNet baseline that outperforms many more complex architectures through careful tuning and post-processing
