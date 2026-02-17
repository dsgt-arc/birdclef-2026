---
slug: "kadircandrisolu/efficientnet-b0-pytorch-inference-birdclef-25"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 710
---

# Kaggle Summary: EfficientNet B0 PyTorch Inference

**Source**: https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-inference-birdclef-25
**Competition**: BirdCLEF 2025
**Score**: Unknown (inference-only notebook)

## Summary

This notebook implements a CPU-based inference pipeline for BirdCLEF 2025 using EfficientNet-B0 with PyTorch. It processes 5-second sliding windows over test soundscapes and supports both single-model and multi-model ensemble predictions with optional test-time augmentation.

## Approach

### Architecture

The model uses EfficientNet-B0 as a backbone from the timm library, configured with single-channel input for grayscale mel spectrograms. The classifier head replaces the default backbone classifier with a linear layer projecting to the number of BirdCLEF 2025 species classes. Global average pooling is applied to backbone features before classification. The model design is generic enough to support other architectures like ResNet by swapping the backbone.

### Data Pipeline

Audio files are loaded at 32kHz sampling rate and segmented into 5-second windows. Each segment is converted to a mel spectrogram using librosa with the following parameters: 1024 FFT size, 512 hop length, 128 mel bins, 50Hz-14kHz frequency range. The mel spectrograms are converted to decibel scale, min-max normalized to [0,1], and resized to 256x256 pixels using OpenCV bilinear interpolation. NaN values in audio are replaced with the mean signal value.

### Training

This notebook only covers inference. The author references separate training notebooks for data preprocessing and model training details.

### Inference / Post-Processing

The inference pipeline processes entire soundscape files by sliding a 5-second window with non-overlapping segments. For each segment, the mel spectrogram is fed through either a single model or an ensemble of models loaded from checkpoint files. When using multiple models, predictions are averaged across all models. Optional test-time augmentation applies horizontal flip (time reversal), vertical flip (frequency reversal), or uses the original spectrogram, with predictions averaged across all TTA variations. The output probabilities use sigmoid activation. Predictions are organized by row_id format (soundscape_name_end_time) to match the competition submission format.

## Key Ideas

- **CPU-optimized inference**: Designed to run on Kaggle's CPU-only notebooks by disabling GPU and keeping batch sizes manageable
- **Flexible ensemble loading**: Automatically discovers all .pth model files in the input directory and supports filtering by fold numbers for k-fold cross-validation ensembles
- **TTA via spectrogram flips**: Implements simple test-time augmentation by flipping spectrograms horizontally (time reversal) and vertically (frequency shift), which are common transformations for audio data
- **Librosa-only preprocessing**: Uses pure librosa for audio processing without PyTorch audio transforms, making the pipeline simpler and more portable
- **Missing species handling**: Checks for missing species columns in the submission and fills them with zeros to ensure compatibility with the sample submission format
- **Adaptive pooling**: Uses AdaptiveAvgPool2d to handle variable feature map sizes from different backbone architectures

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | Not reported |

## Connections

- **Related concepts**: Similar to our experiment 009's inference pipeline using ONNX; this notebook uses PyTorch directly
- **Informs experiments**: The 5-second windowing strategy and mel spectrogram parameters could inform our preprocessing choices
- **Similar approaches**: Shares the ensemble averaging strategy with many top Kaggle solutions; simpler than BamBird/BirdNET teacher-label approaches in exp 010
