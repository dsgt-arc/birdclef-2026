---
slug: "overvalueawareness/bird-25-regnety008-v1-default-v1"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 155
---

# Kaggle Summary: Bird'25 RegNetY008 v1 · default · V1

**Source**: https://www.kaggle.com/code/overvalueawareness/bird-25-regnety008-v1-default-v1
**Competition**: BirdCLEF 2025
**Score**: Unknown (inference-only notebook)

## Summary

This notebook implements a clean inference pipeline for BirdCLEF 2025 using RegNetY-008 as the backbone architecture. It processes 5-second audio segments with mel spectrogram features and supports both single-model and ensemble prediction workflows, with optional test-time augmentation and temporal smoothing post-processing.

## Approach

### Architecture

The model uses RegNetY-008 from the timm library as the backbone, with a custom classification head. The architecture removes the default classifier and replaces it with global average pooling followed by a linear layer mapping to the number of species classes. The model accepts single-channel mel spectrogram inputs and outputs logits for multi-label classification with sigmoid activation during inference.

### Data Pipeline

Audio files are loaded at 32kHz sampling rate and segmented into 5-second windows. Each segment is converted to a mel spectrogram with the following parameters:
- FFT size: 2048
- Hop length: 512
- Mel bins: 128
- Frequency range: 50-14000 Hz
- Power spectrogram (power=2.0)

Spectrograms are converted to decibel scale using reference maximum power, then normalized to [0, 1] range using min-max scaling. The resulting spectrograms are resized to 256x256 pixels using bilinear interpolation. Audio segments shorter than 5 seconds are zero-padded to the required length.

### Training

Training details are not included in this inference-only notebook. The notebook loads pre-trained model checkpoints from a Kaggle model artifact (shionao7/regnrty008). Models are stored with fold-specific naming, allowing selective loading of specific cross-validation folds if desired.

### Inference / Post-Processing

The pipeline supports flexible ensemble prediction across multiple model checkpoints. Test-time augmentation can be enabled with three variations:
1. Original spectrogram (no augmentation)
2. Horizontal flip (time reversal)
3. Vertical flip (frequency reversal)

When TTA is active, predictions from all augmentation variants are averaged. For ensemble models, predictions are averaged across all loaded checkpoints.

Post-processing applies temporal smoothing within each soundscape file. Predictions are grouped by soundscape, and each segment's predictions are blended with its neighbors using weighted averaging:
- First segment: 80% current + 20% next
- Middle segments: 20% previous + 60% current + 20% next
- Last segment: 20% previous + 80% current

This smoothing enforces temporal consistency under the assumption that species presence changes gradually rather than abruptly between consecutive 5-second segments.

## Key Ideas

- **Lightweight backbone**: RegNetY-008 is a relatively small model (~6M parameters) from the RegNet family, designed for efficient inference while maintaining reasonable accuracy
- **Simple feature engineering**: Uses standard mel spectrogram features without exotic preprocessing or learned representations, relying on model capacity to extract discriminative patterns
- **Temporal smoothing**: The post-processing step explicitly models temporal correlation by blending predictions across adjacent segments within each soundscape recording
- **Modular pipeline**: The code is organized as a pipeline class with clearly separated stages (audio loading, feature extraction, prediction, post-processing), making it easy to swap components or adjust parameters
- **Ensemble-friendly design**: Supports loading multiple model checkpoints and averaging their predictions, with optional fold selection for cross-validation ensemble strategies

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | Not reported |
| Private LB | Not reported |

Note: This appears to be a derived/remixed notebook based on earlier work (credits mention shionao7's submission and kumarandatascientist's EfficientNet pipeline). The notebook contains inference code only and does not report validation or leaderboard scores.

## Connections

- **Related concepts**: Could inform ensemble strategies, temporal smoothing approaches, and lightweight model selection
- **Informs experiments**: The temporal smoothing technique could be adapted for soundscape prediction tasks; the modular pipeline design provides a template for inference systems
- **Similar approaches**: RegNet architectures are part of the same design space as ResNets and EfficientNets, offering alternative efficiency-accuracy tradeoffs
