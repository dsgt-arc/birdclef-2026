---
slug: "kumarandatascientist/bird25-onlyin-lb-834-changed-cfg"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 363
---

# Kaggle Summary: Bird25 OnlyIN LB.834 Changed CFG

**Source**: https://www.kaggle.com/code/kumarandatascientist/bird25-onlyin-lb-834-changed-cfg
**Competition**: BirdCLEF 2025
**Score**: 0.834 (LB)

## Summary

This inference notebook builds on the EfficientNetV2-S training pipeline by hideyukizushi, achieving a 0.834 leaderboard score primarily through a configuration change that adopts FocalLossBCE from BirdCLEF 2024's 8th place solution. The author reports strong cross-validation performance (0.962 OOF average across 4 folds) using this loss function instead of standard BCE, combined with temporal smoothing post-processing.

## Approach

### Architecture

The solution uses a timm-based EfficientNetV2-S backbone (in21k_ft_in1k variant) with a global average pooling layer followed by a linear classification head. The model accepts single-channel mel spectrogram input and outputs logits for all species classes. The backbone's final classifier layer is replaced with an identity function, and adaptive pooling ensures fixed-dimension features regardless of input spatial size.

### Data Pipeline

Audio processing converts 5-second segments sampled at 32kHz into mel spectrograms with the following parameters: 1034 FFT bins, 64 hop length, 136 mel bands, 20-16000 Hz frequency range. The mel spectrograms are computed using librosa with slaney normalization and HTK settings, converted to power decibels, then min-max normalized to [0,1]. Each spectrogram is resized to 256x256 via OpenCV linear interpolation. Segments shorter than 5 seconds are zero-padded.

### Training

The key modification is using FocalLossBCE, a weighted combination of binary cross-entropy and focal loss. The loss uses alpha=0.25, gamma=2, with BCE weighted at 0.6 and focal loss at 1.4. This approach was borrowed from BirdCLEF 2024's 8th place solution and reportedly improved validation performance. The author trained using the base preprocessing pipeline from Kadircan İdrisoğlu's public notebooks but applied this custom loss function locally. Training used 4-fold cross-validation with OOF scores ranging from 0.9605 to 0.9652.

### Inference / Post-Processing

Inference processes test soundscapes by sliding a 5-second window across the audio, generating one prediction per segment. The notebook supports optional test-time augmentation with 3 variants (original, horizontal flip, vertical flip) and model ensembling across multiple folds. The final key technique is temporal smoothing: predictions are smoothed across consecutive segments using weighted averaging (0.2, 0.6, 0.2 for middle segments; 0.9, 0.1 for endpoints). This smoothing assumes bird presence changes gradually over time rather than abruptly between 5-second windows.

## Key Ideas

- **Focal loss adaptation**: FocalLossBCE with 0.6/1.4 weighting between BCE and focal components addresses class imbalance more effectively than standard BCE, particularly valuable for rare species
- **Temporal smoothing post-processing**: Averaging predictions across adjacent 5-second segments (0.2-0.6-0.2 weighted scheme) exploits temporal continuity in soundscapes, smoothing out prediction noise
- **Minimal augmentation at inference**: TTA is optional and simple (just spatial flips), suggesting the model generalizes well without extensive test-time transformations
- **Configuration over code**: The notebook emphasizes that changing hyperparameters in the config class (loss function parameters) rather than modifying architecture yielded the performance gain

## Results

| Metric | Value |
|--------|-------|
| CV (OOF) | 0.9622 (avg), 0.9605-0.9652 (range) |
| Public LB | 0.834 |
| Private LB | Unknown |

## Connections

- **Related concepts**: Could inform exploration of focal loss variants for handling BirdCLEF+ taxonomy's multi-class imbalance (206 species with varying prevalence)
- **Informs experiments**: Temporal smoothing technique may be applicable to our soundscape predictions, particularly for experiment 009's inference pipeline
- **Similar approaches**: Based on hideyukizushi's EfficientNet pipeline (bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829) and Kadircan İdrisoğlu's preprocessing notebooks
