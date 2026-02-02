# Experiment 002: Supervised ViT Baseline - Results

## Summary

| Metric | ViT-Base (Ours) | ConvNeXT (Baseline) | % of Baseline |
|--------|-----------------|---------------------|---------------|
| **cmAP** | **0.164** | 0.480 | 34% |
| AUROC | 0.578 | 0.883 | 65% |
| F1 Macro | 0.086 | 0.139 | 62% |
| F1 Micro | 0.095 | 0.121 | 79% |
| Top-1 Acc | 0.112 | 0.251 | 45% |
| Top-5 Acc | 0.547 | 0.330 | 166% ⬆️ |

*Note: ConvNeXT baseline from experiment 001 on HSN dataset (test_5s split)*

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | HSN (Hawaii Soundscapes) |
| Train samples | 5,460 |
| Val/Test samples | 12,000 |
| Classes | 21 |
| Model | ViT-Base-Patch16 (pretrained ImageNet-21k) |
| Parameters | 85.5M |
| Epochs trained | 14 (early stopped) |
| Best epoch | 3 |
| Training time | 37 min 16 sec |
| GPU | Quadro RTX 6000 (24GB) |

## Training Configuration

```yaml
model:
  architecture: vit_base_patch16
  img_size: [128, 500]  # (n_mels, target_frames)
  patch_size: 16
  pretrained: true

training:
  max_epochs: 50
  lr: 1.0e-4
  weight_decay: 0.01
  warmup_epochs: 5
  precision: 16-mixed
  batch_size: 32

data:
  sample_rate: 32000
  target_duration: 5.0
  n_mels: 128
  frame_shift_ms: 10
```

## Training Progression

| Epoch | val/cmap | val/auroc | Status |
|-------|----------|-----------|--------|
| 0 | 0.105 | - | New best |
| 1 | 0.128 | - | New best (+0.022) |
| 2 | - | - | No improvement |
| 3 | **0.164** | - | **New best (+0.037)** |
| 4-13 | - | - | No improvement (early stop triggered) |

## Final Metrics

### Test Set (using best checkpoint from epoch 3)

```
test/cmap:      0.1644
test/auroc:     0.5776
test/f1_macro:  0.0860
test/f1_micro:  0.0952
test/top1_acc:  0.1123
test/top5_acc:  0.5466
test/loss:      0.1369
```

### Training Set (final epoch)

```
train/cmap:     0.984  (overfitting)
train/f1_macro: 0.952
train/loss:     0.010
```

## Analysis

### Observations

1. **Severe overfitting**: Train cmAP (0.984) >> Val cmAP (0.164)
   - Model memorizes training data but doesn't generalize
   - Early stopping triggered after 10 epochs without improvement

2. **Below baseline**: 34% of BirdSet ConvNeXT baseline (0.164 vs 0.480 cmAP)
   - ConvNeXT may be better suited for audio spectrograms
   - ViT may need more data or different augmentation

3. **Top-5 accuracy exceeds baseline**: 54.7% vs 33.0% (166% of baseline)
   - Model often gets correct species in top-5 predictions
   - Primary prediction confidence may be miscalibrated
   - Suggests potential with better calibration/thresholding

### Potential Improvements

1. **Data augmentation**: SpecAugment, time/frequency masking, mixup
2. **Regularization**: Increase dropout, label smoothing
3. **Architecture**: Try Audio Spectrogram Transformer (AST) or ConvNeXT
4. **Learning rate**: Try lower LR or different schedule
5. **Pretraining**: Use AudioSet-pretrained model instead of ImageNet

## Artifacts

| Artifact | Location |
|----------|----------|
| Best checkpoint | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/002-supervised-vit-baseline/checkpoints/best-epochepoch=03.ckpt` |
| W&B run | https://wandb.ai/dsgt-arc/birdset-vit-baseline/runs/dvoxy50z |
| TensorBoard | `outputs/tensorboard/exp002-HSN/` |
| SLURM log | `logs/slurm_3692914.log` |

## Job Info

```
Job ID:        3692914
Node:          atl1-1-03-003-19-0
Wall time:     00:37:16
Memory used:   14.9 GB
Account:       paceship-dsgt_clef2026
```
