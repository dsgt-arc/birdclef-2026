---
slug: "kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 936
---

# Kaggle Summary: EfficientNet B0 Pytorch [Train] | BirdCLEF'25

**Source**: https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25
**Competition**: BirdCLEF 2025
**Score**: Not reported (baseline/starter notebook)

## Summary

A baseline training pipeline for BirdCLEF 2025 using EfficientNet-B0 with PyTorch and timm. The notebook provides a flexible framework supporting both pre-computed spectrograms and on-the-fly audio processing, with 5-fold stratified cross-validation and mixup augmentation. This is designed as a starter notebook with 936 votes, indicating strong community adoption.

## Approach

### Architecture

The model uses EfficientNet-B0 as the backbone, loaded via timm with ImageNet pre-training. The architecture accepts single-channel mel spectrogram input (256x256). The backbone's original classifier head is replaced with a custom linear layer that maps backbone features to the number of BirdCLEF species classes (182 in 2025). Global adaptive average pooling reduces spatial dimensions before classification. The model includes dropout (rate 0.2) and stochastic depth (drop path rate 0.2) for regularization.

### Data Pipeline

Audio files are converted to mel spectrograms using librosa with the following parameters: 32kHz sample rate, 1024 FFT size, 512 hop length, 128 mel bins, and frequency range 50-14000 Hz. Each recording is trimmed or repeated to exactly 5 seconds, then the center 5-second segment is extracted. Spectrograms are resized to 256x256 pixels and normalized to [0,1] range using min-max scaling on the dB-scale mel spectrogram.

The pipeline supports two modes: loading pre-computed spectrograms from disk (faster training) or generating spectrograms on-the-fly during data loading. The notebook includes a companion preprocessing notebook that generates and caches spectrograms as numpy arrays.

Stratified 5-fold cross-validation splits the data by primary species label to ensure balanced class distribution across folds. Multi-label support is included by encoding both primary and secondary labels when available.

**Augmentations** (applied with 50% probability during training):
- Time masking: 1-3 horizontal stripes of width 5-20 pixels
- Frequency masking: 1-3 vertical stripes of height 5-20 pixels
- Brightness/contrast jitter: gain [0.8, 1.2], bias [-0.1, 0.1]

### Training

**Loss function**: Binary cross-entropy with logits (multi-label classification)

**Optimizer**: AdamW with learning rate 5e-4 and weight decay 1e-5

**Scheduler**: Cosine annealing with minimum learning rate 1e-6 over 10 epochs

**Mixup**: Beta distribution with alpha=0.5 applied to both inputs and targets during training

**Batch size**: 32

**Epochs**: 10 (default), 2 in debug mode

**Hardware**: Single GPU (Kaggle P100)

The training loop calculates per-class AUC-ROC averaged across all species as the validation metric. Best model checkpoints are saved per fold based on validation AUC. The notebook includes a debug mode that limits data to 1000 samples and runs for only 2 epochs to enable rapid experimentation.

### Inference / Post-Processing

Inference is handled in a separate companion notebook (not included here). The training notebook saves model checkpoints containing model weights, optimizer state, scheduler state, and training metadata for each fold. No ensemble strategy is implemented in the training code, but the 5-fold structure supports ensemble inference by averaging predictions across fold models.

## Key Ideas

- **Dual-mode data loading**: Support for both cached spectrograms (fast iteration) and on-the-fly generation (lower disk usage) makes the pipeline adaptable to different resource constraints
- **Center cropping strategy**: Extracting the center 5 seconds rather than random cropping assumes bird vocalizations are centered in recordings, which may be valid for this competition's data format
- **Multi-label encoding**: Both primary and secondary species labels are used to create multi-hot target vectors, leveraging additional supervision from background species
- **Mixup with multi-label BCE**: Applies mixup augmentation to both spectrograms and multi-hot label vectors, interpolating targets as well as inputs
- **Spectrogram-level augmentations**: Time and frequency masking simulate occlusions and missing frequency bands, similar to SpecAugment techniques used in speech recognition
- **Per-fold checkpointing**: Saves separate checkpoints for each fold to support both ensemble prediction and fold-specific analysis

## Results

| Metric | Value |
|--------|-------|
| CV (mean AUC across folds) | Not reported |
| Public LB | Not reported |
| Private LB | Not reported |

The notebook is designed as a baseline/starter and does not report specific performance metrics. The author mentions training for 10 epochs and using the resulting weights in an inference notebook, but does not publish validation AUC scores or leaderboard performance.

## Connections

- **Related concepts**:
  - Multi-label classification with secondary species annotations
  - Mixup augmentation for improved generalization
  - Spectrogram augmentation strategies (time/frequency masking)
- **Informs experiments**:
  - Baseline for comparison with Mamba-based architectures (exp 006-009)
  - Reference for mel spectrogram preprocessing parameters (FFT size, hop length, frequency range)
  - Example of dual-mode data loading (cached vs on-the-fly)
- **Similar approaches**:
  - EfficientNet family commonly used in audio classification competitions
  - Center cropping vs random cropping trade-offs
  - Multi-label encoding strategies
