# Results: Frame-Based Tokenization for ViT

**Date Completed**: 2026-02-18
**Author**: Murilo

## Summary

Vertical frame tokenization shows a consistent improvement over standard 16×16 square patches for ViT-Small trained from scratch on HSN. The best config — **128×4 (cmAP=0.118)** — improves +37% over the 16×16 baseline (0.086). However, all cmAP values remain low in absolute terms, and severe overfitting (train cmAP ~0.94 vs test ~0.12) suggests that regularization and data augmentation — not tokenization — are the primary bottleneck.

The streaming baseline (0.086) exceeds exp 003's ViT-Small scratch (0.073) by 18%, and the best vertical config (0.118) is within 2.5% of exp 003's pretrained ViT-Small (0.121), though these comparisons are limited by single-seed runs and low absolute performance.

**Verdict**: confirmed — vertical frame tokenization improves cmAP by 15–37% over square 16×16 patches on HSN when training ViT-Small from scratch. The effect is consistent across all three vertical configs but the absolute performance is still low.

## Observations

### Test Set Results (HSN `test_5s`, 12,000 samples)

The streaming pipeline splits full recordings into all non-overlapping 5s chunks (~12× more training data per epoch: 66,504 chunks vs 5,460 single-crop samples), uses event-aware label filtering, and includes an AMP precision fix for nnAudio mel. All 4 runs landed on RTX 6000 GPUs (matching exp 003 hardware).

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.086 | 0.544 | 0.096 | 0.558 | 0.123 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.099 | 0.633 | 0.496 | 0.607 | 0.106 | 21.54M |
| A: 128×2 | 128×2 | 250 | 0.115 | 0.587 | 0.037 | 0.445 | 0.146 | 21.50M |
| B: 128×4 | 128×4 | 125 | 0.118 | 0.579 | 0.056 | 0.584 | 0.139 | 21.55M |

### Relative Improvement over 16×16 Baseline

| Config | cmAP | cmAP Lift | AUROC Lift |
|--------|------|-----------|------------|
| Baseline (16×16) | 0.086 | — | — |
| C: 128×1 | 0.099 | +15% (1.15×) | +16% |
| A: 128×2 | 0.115 | +34% (1.34×) | +8% |
| B: 128×4 | 0.118 | +37% (1.37×) | +6% |

### Training Dynamics

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 16 | 32 | 0.086 | 3.32 it/s | 1h 39m |
| A: 128×2 | 34 | 50 | 0.115 | 3.43 it/s | 2h 31m |
| B: 128×4 | 22 | 36 | 0.118 | 3.47 it/s | 1h 48m |
| C: 128×1 | 6 | 22 | 0.099 | 2.50 it/s | 1h 28m |

Note: 519 steps/epoch (66,504 chunks ÷ batch 128).

### Qualitative

- **128×4 is the best config** (cmAP=0.118), narrowly ahead of 128×2 (0.115). The shorter 125-token sequence allows more effective attention learning with the available data.
- **128×2 trains the longest**: 50 epochs (best at 34) vs 36 for 128×4, 32 for baseline, 22 for 128×1. The 250-token sequence has enough capacity to keep learning but the model severely overfits (train cmAP=0.940 vs test cmAP=0.115).
- **128×1 converges earliest and stops earliest**: best at epoch 6, early-stopped at 22. 500 tokens is too many for effective attention — the O(n²) cost means the model overfits before it can fully leverage the training data.
- **Baseline streaming (0.086) exceeds exp 003 scratch (0.073)**: the 16×16 baseline benefits from the correct data pipeline (+18% over exp 003).
- **Severe overfitting in all configs**: 128×2 train cmAP=0.940 vs test cmAP=0.115 (8.2× gap); baseline train cmAP=0.496 vs test cmAP=0.086 (5.8× gap). Data augmentation (mixup, SpecAugment) is the most promising next step.
- **All runs on RTX 6000**: all streaming runs correctly landed on RTX 6000, matching exp 003 hardware.
- **Training speed scales inversely with token count**: 128×4 (125 tokens) at 3.47 it/s, 128×2 (250) at 3.43 it/s, baseline (248) at 3.32 it/s, 128×1 (500) at 2.50 it/s.

## Analysis

All three vertical configs outperform the 16×16 baseline by 15–37% in cmAP. The improvement is consistent, but the absolute cmAP values are low across the board (0.086–0.118), indicating that tokenization alone is not sufficient — the model needs better regularization.

**Optimal temporal resolution**: 128×4 (40ms, 125 tokens) is the best config, slightly ahead of 128×2 (20ms, 250 tokens). The shorter 125-token sequence allows more effective self-attention with 6 heads on this dataset. 128×1 (10ms, 500 tokens) is the weakest vertical config — 500 tokens appears to exceed the effective capacity of ViT-Small's attention for this data volume.

**Overfitting is the primary bottleneck**: The massive train-test gap (8× for 128×2) shows that model capacity is not the issue — the model can memorize the training data. Data augmentation (mixup, SpecAugment, time-frequency masking) is likely the most impactful next step, more so than further tokenization experiments.

**Context vs exp 003**: The streaming 16×16 baseline (0.086) exceeds exp 003's ViT-Small scratch (0.073) by 18%, confirming that the streaming pipeline's data processing (non-overlapping chunks from full recordings) is a meaningful improvement over the original data pipeline.

### Confounders

**Single seed**: Each config run once — no variance estimate. The consistent improvement across all 3 vertical configs makes the finding directionally robust, but multi-seed runs are needed to quantify the effect size.

**Hyperparameters tuned for 16×16**: lr=5e-4, wd=0.05, warmup=10 carried over from exp 003. Vertical frame configs may benefit from different hyperparameters.

**Low absolute performance**: All cmAP values are below 0.12. The relative improvements (15–37%) may not generalize to higher-performing models where the baseline is stronger.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16_streaming/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2_streaming/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4_streaming/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1_streaming/HSN/metrics.json` |
| SLURM logs | `logs/slurm_baseline_16x16_streaming_4021520.log` |
| SLURM logs | `logs/slurm_vertical_128x2_streaming_4021521.log` |
| SLURM logs | `logs/slurm_vertical_128x4_streaming_4021522.log` |
| SLURM logs | `logs/slurm_vertical_128x1_streaming_4021523.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/*-streaming/` |
| W&B | [baseline-16x16-streaming](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/wlpbgmge) |
| W&B | [vertical-128x2-streaming](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/zsqbtz1a) |
| W&B | [vertical-128x4-streaming](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/d04oceur) |
| W&B | [vertical-128x1-streaming](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/z0be2ogg) |
| Visualization | `artifacts/tokenization_comparison.png` |
| Earlier runs | [results-earlier-runs.md](results-earlier-runs.md) (superseded — flawed data pipeline) |
| Code | Branch `mg/tokenization` |

## Next Steps

- [x] ~~Torchaudio mel rerun~~ — completed, mel backend confound resolved
- [x] ~~Streaming pipeline + AMP fix~~ — completed
- [ ] Add data augmentation (mixup, SpecAugment) — severe overfitting (train cmAP ~0.94 vs test ~0.12) is the primary bottleneck
- [ ] Tune hyperparameters for vertical frames — shorter warmup (2–5 epochs) and higher LR; also try different dropout/weight decay
- [ ] Test 128×2 or 128×4 vertical frames with pretrained ViT weights (reshape/interpolate the 16×16 patch embedding)
- [ ] Run multi-seed experiments on top configs (128×4, 128×2) to estimate variance
- [ ] Try horizontal frame tokenization (W×500 patches spanning full time axis) as proposed in the proposal's future scope

**Leads to**: Data augmentation to reduce overfitting (likely largest remaining gain); combining vertical frame tokenization with pretrained backbones; hyperparameter tuning for non-square patch ViTs.
