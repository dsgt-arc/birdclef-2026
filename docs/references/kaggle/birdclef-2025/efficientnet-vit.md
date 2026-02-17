---
slug: "vtqunh/efficientnet-vit"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 13
---

# Kaggle Summary: EfficientNet + ViT

**Source**: https://www.kaggle.com/code/vtqunh/efficientnet-vit
**Competition**: BirdCLEF 2025
**Score**: Not reported

## Summary

This notebook implements a dual-backbone ensemble architecture that combines a Vision Transformer (ViT) adapted for audio spectrograms with an EfficientNet CNN. The approach fuses global attention-based features from the transformer with local convolutional features to create a richer representation for multi-label bird species classification.

## Approach

### Architecture

The model uses a two-stream architecture with feature fusion:

1. **Transformer Stream**: ViT-Base with 16x16 patches (768 embedding dimensions, 12 layers, 12 heads) pre-trained on ImageNet. Adapted for single-channel mel spectrograms at 256x256 resolution. Uses 0.2 drop path rate and 0.3 dropout.

2. **CNN Stream**: EfficientNet-B0 pre-trained on ImageNet, also adapted for single-channel input with dropout and drop path regularization.

3. **Fusion Layer**: Concatenates outputs from both backbones, then passes through a linear projection, batch normalization, SiLU activation, and dropout to produce unified features.

4. **Classification Head**: Single linear layer mapping fused features to 182 species classes.

The architecture leverages complementary inductive biases: transformers capture long-range spectro-temporal patterns while CNNs extract local texture and frequency features.

### Data Pipeline

**Audio Processing**:
- Resamples audio to 32kHz
- Generates mel spectrograms: 128 mel bins, 1024 FFT, 512 hop length, 50-14000 Hz frequency range
- Crops/pads to 5 seconds duration
- Resizes spectrograms to 256x256 for model input
- Uses pre-computed spectrograms from an external dataset (thanhtranguyen/birdclef-melspec-data)

**Augmentation** (applied with 50% probability during training):
- Time masking: 1-3 horizontal stripes, 5-20 pixels wide
- Frequency masking: 1-3 vertical stripes, 5-20 pixels tall
- Random brightness/contrast: gain 0.8-1.2x, bias ±0.1
- Mixup: beta distribution with alpha=0.5

**Data Split**: 80/20 train/validation random split from training metadata.

### Training

**Loss Function**: Asymmetric Loss for Multi-Label classification with gamma_neg=4, gamma_pos=1, clip=0.05. This loss downweights easy negatives more aggressively than hard positives, which is beneficial for highly imbalanced multi-label scenarios.

**Mixup**: Applied in the forward pass during training by blending input spectrograms and targets from randomly paired samples.

**Optimizer**: AdamW with learning rate 5e-4, weight decay 1e-5

**Scheduler**: CosineAnnealingLR with T_max=8 epochs, minimum LR 1e-6

**Training Configuration**:
- 8 epochs
- Batch size 24 (notebook shows 24 despite CFG.batch_size=32)
- Hardware: Not specified (GPU enabled in kernel metadata set to false, suggesting CPU-only or oversight)
- Metric: Mean per-class ROC-AUC across all species

**Cross-Validation**: The code includes scaffolding for 5-fold stratified CV but appears to run in debug mode or single-fold mode based on execution.

### Inference / Post-Processing

The notebook does not include inference or submission generation code. Training outputs model checkpoints when validation AUC improves. The prediction pipeline would likely:
- Load pre-computed or generate mel spectrograms for test audio
- Apply both backbones in parallel
- Fuse features and generate logits
- Apply sigmoid to convert to probabilities

No test-time augmentation, ensembling, or temporal aggregation strategies are shown.

## Key Ideas

- **Dual-backbone fusion**: Combines inductive biases of transformers (global context) and CNNs (local patterns) rather than relying on a single architecture family.
- **Pre-computed spectrograms**: Uses an external dataset of pre-generated mel spectrograms to avoid recomputing features during training, improving throughput.
- **Asymmetric loss**: Employs a specialized multi-label loss that handles class imbalance by asymmetrically penalizing false positives vs. false negatives.
- **Mixup augmentation**: Integrated directly into the model's forward pass rather than as a separate data pipeline step, ensuring consistent application.
- **ImageNet pre-training**: Leverages both ViT and EfficientNet weights trained on natural images, demonstrating transfer learning across very different domains (images → audio spectrograms).

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported (training loop prints per-epoch AUC but final results not shown) |
| Public LB | Not available |
| Private LB | Not available |

The notebook focuses on training setup and does not report final validation scores or leaderboard performance.

## Connections

- **Related concepts**:
  - Transfer learning from image models to audio spectrograms
  - Multi-label classification with asymmetric loss functions
  - Ensemble architectures via feature fusion
- **Informs experiments**:
  - Could inform architecture choices for combining different model families (e.g., Mamba + CNN fusion)
  - Asymmetric loss might be useful for handling BirdCLEF's class imbalance
  - Pre-computed spectrogram caching strategy relevant for data pipeline optimization
- **Similar approaches**:
  - Standard ViT and CNN baselines for BirdCLEF
  - Dual-stream architectures common in audio-visual learning
