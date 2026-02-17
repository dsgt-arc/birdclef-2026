---
slug: "hiranorm/lb-0-81-efficientnetv2-b0-pytorch-inference"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 138
---

# Kaggle Summary: EfficientNetV2 B0 Pytorch Inference

**Source**: https://www.kaggle.com/code/hiranorm/lb-0-81-efficientnetv2-b0-pytorch-inference
**Competition**: BirdCLEF 2025
**Score**: 0.81 (Public LB)

## Summary

This inference notebook demonstrates a production-ready pipeline for BirdCLEF 2025 using EfficientNetV2-B0 models trained on mel spectrograms. The approach supports both single-model and ensemble inference with optional test-time augmentation, processing 5-second audio segments independently for temporal localization.

## Approach

### Architecture

The model uses `tf_efficientnetv2_b0.in1k` from the timm library as the backbone, with modifications for single-channel mel spectrogram input. The architecture includes dropout regularization (both standard dropout at 0.2 and stochastic depth/drop path at 0.2) to prevent overfitting. The backbone features are pooled using adaptive average pooling and fed into a linear classifier for multi-label classification across all BirdCLEF species.

### Data Pipeline

Audio files are loaded at 32 kHz sampling rate and split into non-overlapping 5-second windows. Each segment is converted to a mel spectrogram with 128 mel bins, 1024 FFT size, 512 hop length, and frequency range 20-16000 Hz. The power spectrograms are converted to decibel scale and normalized to [0,1] range using min-max normalization. Spectrograms are resized to 256x256 pixels using linear interpolation to match the expected input dimensions.

### Training

Training details are referenced in separate notebooks. The notebook loads pre-trained model checkpoints from disk and supports loading multiple fold models for ensemble inference. Models are configured to run on CPU for Kaggle submission compatibility.

### Inference / Post-Processing

The pipeline processes each test soundscape file sequentially, extracting 5-second segments and generating per-segment predictions. For ensemble inference, predictions from multiple fold models are averaged. Optional test-time augmentation includes three variants: original spectrogram, horizontal flip (time reversal), and vertical flip (frequency inversion). TTA predictions are averaged to produce final scores. The notebook generates row IDs in the format `soundscape_id_end_time` where end_time is the segment boundary in seconds.

## Key Ideas

- Segment-level prediction strategy treats each 5-second window independently, allowing temporal localization of species detections within longer soundscapes
- Min-max normalization of mel spectrograms on a per-segment basis ensures consistent input scaling regardless of recording volume variations
- Support for both single-fold and multi-fold ensemble inference enables flexible model deployment strategies
- Test-time augmentation with spectral flips provides simple augmentation without requiring additional audio processing
- CPU-based inference design prioritizes Kaggle submission compatibility over speed
- The pipeline uses sigmoid activation for multi-label prediction, producing independent probabilities for each species
- Adaptive global average pooling collapses spatial dimensions, making the architecture agnostic to minor spectrogram shape variations

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.81 |
| Private LB | Not reported |

## Connections

- **Related concepts**: This approach demonstrates standard mel spectrogram preprocessing and EfficientNet transfer learning for audio classification
- **Informs experiments**: The segment-level prediction strategy and mel spectrogram parameters could inform baseline comparisons
- **Similar approaches**: Represents the mainstream ConvNet + mel spectrogram paradigm common in BirdCLEF competitions
