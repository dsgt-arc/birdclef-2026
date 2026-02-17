---
slug: "hideyukizushi/bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 401
---

# Kaggle Summary: EfficientNetV2-S with FocalLossBCE

**Source**: https://www.kaggle.com/code/hideyukizushi/bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829
**Competition**: BirdCLEF 2025
**Score**: CV 0.962, Public LB 0.829

## Summary

This inference-only notebook loads locally trained EfficientNetV2-S models and applies them to test soundscapes. The key contribution is demonstrating that a hybrid focal loss combining BCE with focal weighting (borrowed from BirdCLEF 2024 8th place solution) significantly improves cross-validation performance compared to standard BCE loss, achieving 0.962 CV across 4 folds.

## Approach

### Architecture

The model uses `tf_efficientnetv2_s.in21k_ft_in1k` from the timm library as the backbone, pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k. The architecture consists of:
- EfficientNetV2-S backbone with classifier head removed
- Adaptive average pooling to handle variable feature map sizes
- Single linear classification head for multi-label prediction

Single-channel input spectrograms are passed to the model (1 channel rather than 3-channel RGB).

### Data Pipeline

Audio processing converts soundscapes into mel spectrograms using librosa with the following parameters:
- Sampling rate: 32kHz
- Window size: 5 seconds per segment
- FFT size: 2048, hop length: 512
- Mel bins: 512, frequency range: 20-16000 Hz
- Power spectrogram with Slaney normalization and HTK formula
- Power-to-dB conversion followed by min-max normalization
- Final resize to 256x256 pixels using OpenCV bilinear interpolation

Each test soundscape is split into consecutive 5-second segments. Segments shorter than 5 seconds are zero-padded.

### Training

The model was trained locally (not in this inference notebook) using:
- **FocalLossBCE**: A weighted combination of BCE loss (weight 0.6) and sigmoid focal loss (weight 1.4)
- Focal loss parameters: alpha=0.25, gamma=2
- 4-fold cross-validation with ensemble inference

The focal loss formulation addresses class imbalance by down-weighting easy examples and focusing on hard negatives, while the BCE component provides stable gradient signals.

### Inference / Post-Processing

The inference pipeline includes:
1. Load ensemble of 4 models (one per fold)
2. Process each 5-second segment independently
3. Average predictions across all models in the ensemble
4. Apply temporal smoothing: each segment's prediction is blended with neighboring segments using weights [0.2, 0.6, 0.2] for [previous, current, next]. Edge segments use [0.9, 0.1] weighting.

Optional test-time augmentation (TTA) is available but disabled in the final submission. TTA options include horizontal flip (time reversal) and vertical flip (frequency reversal).

## Key Ideas

- **Hybrid loss function**: Combining BCE (60%) and focal loss (140%) balances stable training with hard example mining, achieving much higher CV than BCE alone
- **Temporal smoothing**: Simple weighted average across consecutive 5-second segments smooths prediction noise, assuming bird presence is temporally correlated
- **EfficientNetV2-S**: Smaller than EfficientNet-B models but benefits from improved training efficiency and ImageNet-21k pre-training
- **Single-channel spectrograms**: Uses grayscale mel spectrograms rather than 3-channel RGB, reducing model input dimensions
- **High-resolution mel**: 512 mel bins provide fine frequency resolution despite downsampling to 256px height

## Results

| Metric | Value |
|--------|-------|
| Fold 0 CV | 0.9652 |
| Fold 1 CV | 0.9605 |
| Fold 2 CV | 0.9607 |
| Fold 3 CV | 0.9626 |
| OOF CV | 0.9622 |
| Public LB | 0.829 |

The large gap between CV (0.962) and LB (0.829) suggests potential overfitting to the local validation split or domain shift in test soundscapes.

## Connections

- **Related concepts**: This approach builds on standard CNN-based spectrogram classification, similar to computer vision transfer learning workflows
- **Informs experiments**: The FocalLossBCE formulation could be applied to SSAMBA or other architectures to address class imbalance in BirdCLEF datasets
- **Similar approaches**: Based on Kadircan İdrisoğlu's baseline pipeline (preprocessing, training, inference structure), with loss function modification from BirdCLEF 2024 8th place solution
