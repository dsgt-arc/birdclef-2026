# Proposal: Model Training (ConvNeXT-Tiny & ViT-Small)

**Status**: proposed
**Created**: 2026-02-08
**Author**: Murilo

## Hypothesis

Training ConvNeXT-Tiny and ViT-Small from scratch on HSN will produce substantially lower cmAP than pretrained variants, quantifying the pretrain advantage. ConvNeXT-Tiny pretrained should approach exp001's cmAP=0.480, and both scratch models will underperform exp002's ViT-Base pretrained (cmAP=0.164).

## Background

Experiment 002 trained ViT-Base (85.5M params) with ImageNet pretrained weights on HSN, achieving cmAP=0.164 (34% of BirdSet baseline). Experiment 001 ran inference with a pre-trained ConvNeXT from `bioacoustics_model_zoo` (cmAP=0.480). Comparing a scratch-trained ViT against a pre-trained ConvNeXT is unfair. This experiment trains both architectures from scratch and with pretrained weights to establish fair, size-comparable baselines.

## Method

### Approach

Train 4 models in a 2x2 matrix: {ConvNeXT-Tiny, ViT-Small} x {scratch, pretrained}. Use a generic `AudioClassifier` class that handles 1-channel input adaptation for both model families.

### Setup

| Component | Details |
|-----------|---------|
| Data | HSN dataset from BirdSet, mel-spectrograms (128x500) |
| Compute | 1x RTX 6000 GPU, 64GB RAM, 8 CPUs, up to 12h per run |
| Dependencies | timm>=0.9.0, PyTorch Lightning, W&B |
| Code | `user/murilo/experiments/003-model-training/` |

### Procedure

1. Create `AudioClassifier` supporting both ViT and ConvNeXT families
2. Train all 4 configurations on HSN
3. Compare cmAP across all variants

### Variables

- **Independent**: Architecture (ConvNeXT-Tiny vs ViT-Small), initialization (scratch vs pretrained)
- **Dependent**: cmAP, training curves, convergence speed
- **Controlled**: Dataset (HSN), input size (128x500), training infrastructure

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| cmAP | Class-mean average precision (primary) |
| val/loss | Validation loss (convergence indicator) |

### Baseline

| Run | cmAP |
|-----|------|
| Exp 001 - ConvNeXT (bioacoustics_model_zoo) | 0.480 |
| Exp 002 - ViT-Base pretrained | 0.164 |

### Success Criteria

- **Confirm if**: Pretrained variants achieve higher cmAP than scratch variants for both architectures
- **Reject if**: Scratch variants match or exceed pretrained variants (would suggest pretraining is not helpful)

## Limitations

- Only testing on HSN dataset; generalization to XCM not tested
- Scratch hyperparameters (lr, warmup, epochs) are educated guesses; may need tuning
- ConvNeXT-Tiny (~28.6M) and ViT-Small (~22M) are similar but not identical in parameter count
