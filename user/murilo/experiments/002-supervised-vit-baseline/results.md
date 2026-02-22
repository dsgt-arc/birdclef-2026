# Experiment 002: Supervised ViT Baseline - Results

**Date Completed**: 2026-02-21
**Author**: Murilo

## Summary

| Metric | ViT-Base (Ours) | ConvNeXT (Baseline) | % of Baseline |
|--------|-----------------|---------------------|---------------|
| **cmAP** | **0.186** | 0.480 | 39% |
| AUROC | 0.662 | 0.883 | 75% |
| F1 Macro | 0.092 | 0.139 | 66% |
| F1 Micro | 0.070 | 0.121 | 58% |
| Top-1 Acc | 0.100 | 0.251 | 40% |
| Top-5 Acc | 0.222 | 0.330 | 67% |

*ConvNeXT baseline from experiment 001 on HSN dataset (test_5s split)*

This run uses the streaming data pipeline (`StreamingBirdSetDataModule`): non-overlapping 5s chunks from full recordings (~66,504 training chunks vs 5,460 center-crop samples in the original pipeline). cmAP improved from 0.164 to 0.186 (+13%).

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | HSN (Hawaii Soundscapes) |
| Train chunks | 66,504 (from 5,460 recordings) |
| Val/Test samples | 12,000 |
| Classes | 21 |
| Model | ViT-Base-Patch16 (pretrained ImageNet-21k) |
| Parameters | 85.5M |
| Data pipeline | StreamingBirdSetDataModule (non-overlapping 5s chunks) |
| Epochs trained | 20 (early stopped at epoch 9) |
| Best epoch | 9 |
| Training time | 1h 43m 49s |
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
  streaming: true
```

## Training Progression

| Epoch | val/cmap | Status |
|-------|----------|--------|
| 0 | 0.185 | New best |
| 1-8 | — | No improvement |
| 9 | **0.186** | **New best (+0.001)** |
| 10-19 | — | No improvement (early stop triggered) |

## Final Metrics

### Test Set (using best checkpoint from epoch 9)

```
test/cmap:      0.1861
test/auroc:     0.6619
test/f1_macro:  0.0924
test/f1_micro:  0.0703
test/top1_acc:  0.1004
test/top5_acc:  0.2222
test/loss:      0.1540
```

## Streaming vs Center-Crop Comparison

| Metric | Center-Crop | Streaming | Change |
|--------|------------|-----------|--------|
| cmAP | 0.164 | 0.186 | +13% |
| AUROC | 0.578 | 0.662 | +15% |
| F1 Macro | 0.086 | 0.092 | +7% |
| Top-1 Acc | 0.112 | 0.100 | -11% |
| Top-5 Acc | 0.547 | 0.222 | -59% |
| Best Epoch | 3 | 9 | Later convergence |

The streaming pipeline improves cmAP and AUROC but reduces top-k accuracy. The center-crop run's anomalously high top-5 accuracy (0.547) was likely an artifact of the limited training data causing the model to predict a narrow set of species with high confidence.

## Analysis

### Observations

1. **Improved over center-crop**: Streaming pipeline cmAP (0.186) beats center-crop (0.164) by +13%, confirming that more training data helps ViT-Base
2. **Still below baseline**: 39% of BirdSet ConvNeXT baseline (0.186 vs 0.480 cmAP)
3. **Similar to exp 003 ConvNeXT-Tiny pretrained**: Despite having 3x more parameters (85.5M vs 27.8M), ViT-Base pretrained (0.186) slightly trails ConvNeXT-Tiny pretrained (0.191)
4. **Overfitting persists**: Model converges quickly (best at epoch 9 out of 50) and then stops improving, suggesting the model capacity still exceeds what the data can support

### Key Takeaway

ViT-Base pretrained with ImageNet-21k achieves 0.186 cmAP on HSN, comparable to ConvNeXT-Tiny pretrained (0.191) but with 3x more parameters. This confirms that architecture efficiency matters — ConvNeXT's inductive bias for spectrograms is more parameter-efficient than ViT's general-purpose attention.

## Artifacts

| Artifact | Location |
|----------|----------|
| Best checkpoint | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/002-supervised-vit-baseline/checkpoints/best-epochepoch=09.ckpt` |
| W&B run | [exp002-HSN (streaming)](https://wandb.ai/dsgt-arc/birdset-vit-baseline/runs/fh21wppu) |
| TensorBoard | `outputs/tensorboard/exp002-HSN/` |
| SLURM log | `logs/slurm_4100662.log` |
| Earlier run | [results-earlier-runs.md](results-earlier-runs.md) (superseded — center-crop data pipeline) |

## Job Info

```
Job ID:        4100662
Node:          atl1-1-02-004-29-0
Wall time:     01:43:49
Memory used:   10.7 GB
Account:       paceship-dsgt_clef2026
```

## Next Steps

- [x] ~~Re-run with streaming data pipeline~~ → completed, results above
- [x] ~~Try smaller architectures~~ → exp 003 (ConvNeXT-Tiny, ViT-Small)
- [x] ~~Explore audio-specific pretrained backbones~~ → exp 005 (EAT/BEATs)
- [ ] Data augmentation to address overfitting
- [ ] Compare with audio-pretrained encoders (exp 005)

**Leads to**: Exp 003 (smaller architectures comparison), exp 005 (audio-domain pretrained encoders).
