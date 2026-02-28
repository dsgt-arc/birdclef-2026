# Results: EfficientNet-B1 Linear Probe

**Date Completed**: 2026-02-18
**Author**: anthony

## Summary

A frozen EfficientNet-B1 pretrained on BirdSet-XCL achieves **0.315 cmAP** with
a linear probe, confirming that backbone feature quality is the primary bottleneck
in our SSAMBA-Tiny pipeline (0.179 cmAP with full finetuning).

**Verdict**: confirmed

## Observations

### Quantitative

| Metric | SSAMBA-Tiny (exp 007) | EfficientNet-B1 (exp 012) | Delta |
|--------|----------------------|---------------------------|-------|
| val/cmap | 0.179 | **0.315** | +76% relative |
| val/auroc | — | 0.890 | — |
| train/cmap | >0.95 | 0.298 | — |
| Trainable params | 7M (100%) | 264K (3.9%) | -96% |

Training progression (selected epochs):

| Epoch | val/cmap | val/auroc | train/loss | train/cmap |
|-------|----------|-----------|------------|------------|
| 0 | 0.141 | 0.800 | 0.0109 | 0.035 |
| 5 | 0.266 | 0.876 | 0.0072 | 0.195 |
| 10 | 0.295 | 0.885 | 0.0070 | 0.247 |
| 15 | 0.309 | 0.887 | 0.0069 | 0.260 |
| 20 | 0.313 | 0.888 | 0.0068 | 0.285 |
| 25 | 0.315 | 0.890 | 0.0068 | 0.292 |
| 29 | 0.315 | 0.890 | 0.0067 | 0.298 |

### Qualitative

- **Fast initial convergence**: 0.266 cmAP by epoch 5, then slow improvement to 0.315
- **No overfitting**: val/cmap (0.315) slightly exceeds train/cmap (0.298), likely because
  SpecAugment degrades training-time metrics while improving generalization
- **Plateau from epoch 23**: val/cmap reached 0.315 and did not improve further; early
  stopping (patience=10) was not triggered because the metric never degraded
- Checkpoint timestamps show ~11 min/epoch, ~5h 18m total wall time

## Analysis

The hypothesis is strongly confirmed. With only 3.9% of parameters trainable, the
EfficientNet-B1 BirdSet-XCL backbone produces features that a linear head can
exploit for 0.315 cmAP — 76% higher than our best SSAMBA-Tiny result with full
finetuning. This means the gap between SSAMBA and competitive baselines is
primarily a representation quality problem, not a training recipe or pooling
strategy issue (as also shown by exp 008).

Relative to known benchmarks:

| Model | val/cmap | Kaggle public | Kaggle private | Context |
|-------|----------|---------------|----------------|---------|
| SSAMBA-Tiny from scratch (exp 006) | 0.086 | — | — | No pretraining |
| SSAMBA-Tiny pretrained (exp 007) | 0.179 | — | — | AudioSet init |
| SSAMBA SSL pipeline (exp 009) | 0.425 | 0.484 | 0.476 | SSL pretrain + finetune |
| EfficientNet-B1 v1 (exp 012, n_mels=128) | 0.315 | 0.590 | 0.583 | BirdSet-XCL frozen |
| **EfficientNet-B1 v2 (exp 012, n_mels=256)** | **0.336** | **0.615** | **0.649** | **BirdSet-XCL frozen, native mel** |
| BirdSet HSN benchmark (EfficientNet) | 0.35–0.38 | — | — | Full finetune, n_mels=256 |
| ConvNeXt (murilo/exp 001) | 0.48 | — | — | ImageNet pretrained |
| HawkEars baseline | — | 0.778 | 0.810 | Reference |
| BirdSet HSN SOTA (Bird-MAE ViT-L) | ~0.55 | — | — | Foundation model |

The v2 result (n_mels=256, matching BirdSet training resolution) scores 0.649 private
on Kaggle — a +6.6pp improvement over v1 (0.583). The mel resolution fix accounts for
most of this gap, confirming the n_mels mismatch was a meaningful confounder.
Full finetuning could push closer to the 0.35–0.38 val/cmap benchmark range.

### Confounders

- **n_mels mismatch**: BirdSet trained with n_mels=256; we used 128 for consistency
  with the SSAMBA pipeline. This likely underestimates the backbone's true capability.
- **Input channel handling**: BirdSet EfficientNet expects 1-channel mel (not 3-channel
  RGB), which we correctly handle — but this differs from standard ImageNet-pretrained
  EfficientNets that expect 3 channels.
- **Dataset difference**: BirdSet HSN benchmarks use a different eval split. Our
  BirdCLEF 2025 val split (10% of focal recordings) may not be directly comparable.

## Artifacts

| Type | Location |
|------|----------|
| SLURM log | `logs/finetune-4029998.out` |
| Best checkpoint | `logs/finetune/checkpoints/finetune-epoch=27-val/` |
| Last checkpoint | `logs/finetune/checkpoints/last.ckpt` (33 MB) |
| TensorBoard | `logs/finetune/lightning_logs/version_4029998/` |
| Overfit test log | `logs/finetune-4029983.out` |
| Fast dev run log | `logs/finetune-4029959.out` |
| Code | commit `8e18893` |

## Next Steps

- [ ] Full finetuning (unfreeze backbone) to measure ceiling vs linear probe
- [x] Test n_mels=256 to match BirdSet training resolution (v2: val/cmap=0.336, +2.1pp over v1)
- [ ] Use as teacher/baseline for distillation into SSAMBA or lighter architectures
- [x] Kaggle submission with EfficientNet to establish leaderboard baseline (v2: 0.615 public / 0.649 private)

**Leads to**: potential exp 013 (EfficientNet full finetune), distillation concept
