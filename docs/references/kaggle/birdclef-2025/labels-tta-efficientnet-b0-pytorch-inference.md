---
slug: "salmanahmedtamu/labels-tta-efficientnet-b0-pytorch-inference"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 159
---

# Kaggle Summary: Labels TTA EfficientNet B0 Pytorch Inference

**Source**: https://www.kaggle.com/code/salmanahmedtamu/labels-tta-efficientnet-b0-pytorch-inference
**Competition**: BirdCLEF 2025
**Score**: 0.784 (based on referenced base notebook)

## Summary

This inference notebook applies test-time augmentation (TTA) to an EfficientNet-B0 model for bird species detection in soundscapes. The approach ensembles predictions across multiple trained folds and applies temporal smoothing to reduce prediction jitter between consecutive time segments.

## Approach

### Architecture

The model uses an EfficientNet-B0 backbone from the timm library with single-channel input for grayscale mel spectrograms. The backbone's classifier layer is replaced with a custom linear head for multi-label classification across all bird species in the taxonomy. Global average pooling is applied to the feature maps before the final classification layer.

### Data Pipeline

Audio files are processed at 32 kHz sample rate and split into non-overlapping 5-second windows. Each segment is converted to a mel spectrogram using librosa with the following parameters: 1024 FFT size, 512 hop length, 148 mel bins, frequency range 50-14000 Hz. The resulting spectrograms are normalized to 0-1 range using min-max scaling and resized to 256x256 pixels using linear interpolation. NaN values in audio are replaced with the mean signal value.

### Training

Training details are not included in this inference notebook, but the notebook loads pre-trained model checkpoints from a separate dataset. The notebook supports loading multiple fold models for ensemble prediction.

### Inference / Post-Processing

The inference pipeline includes several techniques to improve prediction stability:

1. **Test-Time Augmentation**: Three variations are applied to each spectrogram - original, horizontal flip (time reversal), and vertical flip (frequency inversion). Predictions are averaged across all TTA variants.

2. **Multi-Fold Ensemble**: Multiple model checkpoints are loaded and their predictions are averaged. The notebook can either use all available models or a specific subset of folds.

3. **Temporal Smoothing**: After initial predictions, a weighted averaging filter is applied across consecutive time segments. Middle segments use a 0.2/0.6/0.2 weighting scheme (previous/current/next), while boundary segments use 0.8/0.2 weights. This reduces prediction jitter and creates smoother transitions between time windows.

4. **CPU Inference**: The notebook runs entirely on CPU without GPU acceleration, making it suitable for Kaggle's inference environment.

## Key Ideas

- **TTA Strategy**: Using horizontal flip (time reversal) and vertical flip (frequency shift) as augmentation variants provides additional prediction diversity without significantly increasing inference time.

- **Temporal Smoothing**: The weighted averaging across consecutive 5-second segments addresses a common issue in soundscape classification where abrupt changes in predictions between adjacent windows create unrealistic boundaries.

- **Flexible Model Loading**: The notebook automatically discovers and loads all available model checkpoints, with optional filtering by fold numbers, making it easy to experiment with different ensemble configurations.

- **Spectrogram Parameters**: The choice of 148 mel bins and 50-14000 Hz frequency range targets the typical range of bird vocalizations while excluding very low frequency noise.

## Results

| Metric | Value |
|--------|-------|
| CV     | Unknown |
| Public LB | 0.784 (from referenced base notebook) |
| Private LB | Unknown |

## Connections

- **Related concepts**: Test-time augmentation, temporal smoothing for audio classification, multi-fold ensembling
- **Informs experiments**: Could inform post-processing strategies for our SSAMBA-based predictions, particularly the temporal smoothing approach
- **Similar approaches**: This represents a baseline CNN approach compared to our state-space model experiments
