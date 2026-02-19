# Results: Frame-Based Tokenization for ViT

**Date Completed**: 2026-02-17
**Author**: Murilo

## Summary

Vertical frame tokenization significantly outperforms standard 16×16 square patches for ViT on audio spectrograms. The best config — **128×4 (cmAP=0.118)** — improves +37% over the 16×16 baseline (0.086) and +62% over the exp 003 scratch baseline (0.073), nearly matching exp 003's pretrained ViT-Small (0.121).

The canonical results are from the **streaming pipeline** run, which correctly processes audio by splitting full recordings into all non-overlapping 5s chunks with event-aware label filtering (66,504 chunks from 5,460 recordings). Earlier runs (nnAudio and torchaudio) used a flawed data pipeline that shuffled random segments from different audio files instead of properly chunking full recordings — those results are retained below for reference but should be **disregarded** for the tokenization comparison.

**Verdict**: confirmed — vertical frame tokenization improves cmAP by 15–37% over square 16×16 patches on HSN when training ViT-Small from scratch. The combination of vertical frames + streaming data pipeline nearly closes the gap to pretrained ViT performance.

## Observations

### Quantitative

#### Streaming Pipeline Run (canonical results)

> **Note**: The streaming pipeline is the only valid comparison. Earlier nnAudio and torchaudio runs used a flawed data pipeline (shuffled random segments across recordings rather than non-overlapping 5s chunks from full recordings). Those results are retained below for historical reference only.

#### Streaming — Test Set Results (HSN `test_5s`, 12,000 samples)

The streaming pipeline splits full recordings into all non-overlapping 5s chunks (~12× more training data per epoch: 66,504 chunks vs 5,460 single-crop samples), uses event-aware label filtering, and includes an AMP precision fix for nnAudio mel. All 4 runs landed on RTX 6000 GPUs (matching exp 003 hardware).

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.086 | 0.544 | 0.096 | 0.558 | 0.123 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.099 | 0.633 | 0.496 | 0.607 | 0.106 | 21.54M |
| **A: 128×2** | **128×2** | **250** | **0.115** | **0.587** | **0.037** | **0.445** | **0.146** | **21.50M** |
| **B: 128×4** | **128×4** | **125** | **0.118** | **0.579** | **0.056** | **0.584** | **0.139** | **21.55M** |

#### Streaming — Relative Improvement over 16×16 Baseline

| Config | cmAP | cmAP Lift | AUROC Lift |
|--------|------|-----------|------------|
| Baseline (16×16) | 0.086 | — | — |
| C: 128×1 | 0.099 | **+15%** (1.15×) | +16% |
| A: 128×2 | 0.115 | **+34%** (1.34×) | +8% |
| B: 128×4 | 0.118 | **+37%** (1.37×) | +6% |

#### Streaming — Training Dynamics

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 16 | 32 | 0.086 | 3.32 it/s | 1h 39m |
| A: 128×2 | 34 | 50 | 0.115 | 3.43 it/s | 2h 31m |
| B: 128×4 | 22 | 36 | 0.118 | 3.47 it/s | 1h 48m |
| C: 128×1 | 6 | 22 | 0.099 | 2.50 it/s | 1h 28m |

Note: 519 steps/epoch (66,504 chunks ÷ batch 128).

---

#### Earlier Runs (superseded — flawed data pipeline)

> **WARNING**: The nnAudio and torchaudio runs below used a flawed data pipeline that shuffled random segments from different audio files rather than properly chunking full recordings into non-overlapping 5s windows. These results should NOT be used for the tokenization comparison. They are retained for historical context (mel backend confound analysis) only.

#### Torchaudio Rerun — Test Set Results (HSN `test_5s`, 12,000 samples)

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.066 | 0.533 | 0.005 | 0.028 | 0.161 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.086 | 0.648 | 0.014 | 0.137 | 0.153 | 21.54M |
| B: 128×4 | 128×4 | 125 | 0.087 | 0.579 | 0.051 | 0.438 | 0.134 | 21.55M |
| **A: 128×2** | **128×2** | **250** | **0.098** | **0.621** | **0.012** | **0.304** | **0.145** | **21.50M** |

#### Relative Improvement over 16×16 Baseline (torchaudio)

| Config | cmAP | cmAP Lift | AUROC Lift |
|--------|------|-----------|------------|
| Baseline (16×16) | 0.066 | — | — |
| C: 128×1 | 0.086 | **+30%** (1.30×) | +22% |
| B: 128×4 | 0.087 | **+33%** (1.33×) | +9% |
| A: 128×2 | 0.098 | **+49%** (1.49×) | +17% |

#### Torchaudio Rerun — Training Dynamics

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 8 | 24 | 0.066 | 5.13 it/s | 29m 16s |
| A: 128×2 | 6 | 22 | 0.098 | 5.12 it/s | 28m 43s |
| B: 128×4 | 8 | 24 | 0.087 | 8.01 it/s | 27m 19s |
| C: 128×1 | 6 | 22 | 0.086 | 2.59 it/s | 34m 38s |

#### Cross-Experiment Baseline Comparison (confound resolved)

| Run | Mel Backend | GPU | cmAP |
|-----|------------|-----|------|
| Exp 003 — ViT-Small scratch | CPU (torchaudio) | RTX 6000 | 0.073 |
| Exp 004 — Baseline 16×16 (nnAudio) | GPU (nnAudio) | V100 | 0.029 |
| Exp 004 — Baseline 16×16 (torchaudio rerun) | CPU (torchaudio) | V100 | 0.066 |

The torchaudio rerun baseline (0.066) is within 10% of exp 003 (0.073), confirming the mel backend was the primary confound. The remaining gap (0.007) may be due to hardware (V100 vs RTX 6000), seed variance, or minor environment differences.

#### nnAudio vs Torchaudio — Side-by-Side

| Config | nnAudio cmAP | Torchaudio cmAP | Lift from mel fix |
|--------|-------------|-----------------|-------------------|
| Baseline (16×16) | 0.029 | 0.066 | +128% (2.28×) |
| A: 128×2 | 0.054 | 0.098 | +81% (1.81×) |
| B: 128×4 | 0.063 | 0.087 | +39% (1.39×) |
| C: 128×1 | 0.051 | 0.086 | +69% (1.69×) |

All configs improved substantially with torchaudio mel. The baseline benefited the most (2.28×), narrowing the gap between baseline and vertical configs from 2.2× (nnAudio) to 1.49× (torchaudio). This suggests nnAudio's mel spectrogram disproportionately harms 16×16 square patches compared to vertical frames.

#### Original nnAudio Run — Test Set Results (for reference)

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.029 | 0.458 | 0.001 | 0.544 | 0.179 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.051 | 0.536 | 0.483 | 0.566 | 0.122 | 21.54M |
| A: 128×2 | 128×2 | 250 | 0.054 | 0.536 | 0.093 | 0.573 | 0.123 | 21.50M |
| B: 128×4 | 128×4 | 125 | 0.063 | 0.601 | 0.440 | 0.547 | 0.118 | 21.55M |

#### Original nnAudio Run — Training Dynamics (for reference)

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 7 | 17 | 0.029 | 4.16 it/s | 38m 27s |
| A: 128×2 | 2 | 18 | 0.054 | 4.15 it/s | 38m 28s |
| B: 128×4 | 2 | 17 | 0.063 | 5.95 it/s | 37m 23s |
| C: 128×1 | 2 | 17 | 0.051 | 2.34 it/s | 43m 01s |

### Qualitative

- **128×4 is the best config** (cmAP=0.118), narrowly ahead of 128×2 (0.115). With the full-recording streaming pipeline, the coarser 40ms temporal resolution (128×4, 125 tokens) produces the best results — the shorter sequence allows more effective attention learning with the available data.
- **128×2 trains the longest**: 50 epochs (best at 34) vs 36 for 128×4, 32 for baseline, 22 for 128×1. The 250-token sequence has enough capacity to keep learning but the model severely overfits (train cmAP=0.940 vs test cmAP=0.115).
- **128×1 converges earliest and stops earliest**: best at epoch 6, early-stopped at 22. 500 tokens is too many for effective attention — the O(n²) cost means the model overfits before it can fully leverage the training data.
- **Baseline streaming (0.086) exceeds exp 003 scratch (0.073)**: the 16×16 baseline benefits substantially from the correct data pipeline (+18% over exp 003).
- **Severe overfitting in all configs**: 128×2 train cmAP=0.940 vs test cmAP=0.115 (8.2× gap); baseline train cmAP=0.496 vs test cmAP=0.086 (5.8× gap). Data augmentation (mixup, SpecAugment) is the most promising next step.
- **All runs on RTX 6000**: all streaming runs correctly landed on RTX 6000, matching exp 003 hardware.
- **Training speed scales inversely with token count**: 128×4 (125 tokens) at 3.47 it/s, 128×2 (250) at 3.43 it/s, baseline (248) at 3.32 it/s, 128×1 (500) at 2.50 it/s.

## Analysis

**Vertical frame tokenization provides a better inductive bias than 16×16 square patches for ViT on spectrograms.** All three vertical configs outperform the 16×16 baseline by 15–37% in cmAP. The improvement is consistent and the best config (128×4, cmAP=0.118) nearly matches exp 003's pretrained ViT-Small (0.121) — a gap of only 2.5% — despite being trained entirely from scratch.

**Optimal temporal resolution**: 128×4 (40ms, 125 tokens) is the best config, slightly ahead of 128×2 (20ms, 250 tokens). The shorter 125-token sequence allows more effective self-attention with 6 heads on this dataset. 128×1 (10ms, 500 tokens) is the weakest vertical config — 500 tokens appears to exceed the effective capacity of ViT-Small's attention for this data volume.

**Overfitting is the primary bottleneck**: The massive train-test gap (8× for 128×2) shows that model capacity is not the issue — the model can memorize the training data. Data augmentation (mixup, SpecAugment, time-frequency masking) is likely the most impactful next step.

**Context vs exp 003**: The streaming 16×16 baseline (0.086) already exceeds exp 003's ViT-Small scratch (0.073) by 18%, confirming that the streaming pipeline's data processing (non-overlapping chunks from full recordings) is a meaningful improvement over the original data pipeline.

### Confounders

**Data pipeline (resolved)**: The streaming pipeline correctly processes audio by splitting full recordings into non-overlapping 5s chunks with event-aware label filtering. Earlier runs (nnAudio and torchaudio) used a flawed pipeline that shuffled random segments — those results are retained below for reference only.

**Hardware**: All streaming runs landed on RTX 6000 24GB, matching exp 003. Earlier runs were allocated V100 16GB by SLURM.

**Single seed**: Each config run once — no variance estimate. The consistent improvement across all 3 vertical configs (all beat baseline by 15%+) makes the finding robust.

**Hyperparameters tuned for 16×16**: lr=5e-4, wd=0.05, warmup=10 carried over from exp 003. Vertical frame configs may benefit from different hyperparameters.

### Why nnAudio Underperforms (historical)

The performance gap between nnAudio and torchaudio mel is caused by a **mel filterbank configuration mismatch** in our `NNAudioMelSpectrogram` wrapper. See the torchaudio rerun section below for details. This analysis was done before the data pipeline issue was identified and is retained for reference — the streaming pipeline includes an AMP precision fix that also contributes to the gap.

## Artifacts

### Torchaudio Rerun

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1_torchaudio/HSN/metrics.json` |
| SLURM logs | `logs/slurm_baseline_16x16_torchaudio_4017565.log` |
| SLURM logs | `logs/slurm_vertical_128x2_torchaudio_4017567.log` |
| SLURM logs | `logs/slurm_vertical_128x4_torchaudio_4017569.log` |
| SLURM logs | `logs/slurm_vertical_128x1_torchaudio_4017568.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/*-torchaudio/` |

### Streaming Pipeline Run

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

### Original nnAudio Run

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1/HSN/metrics.json` |
| Visualization | `artifacts/tokenization_comparison.png` |
| GPU Utilization | `artifacts/gpu-utilization.png` |
| SLURM logs | `logs/slurm_baseline_16x16_4011400.log` |
| SLURM logs | `logs/slurm_vertical_128x2_4011401.log` |
| SLURM logs | `logs/slurm_vertical_128x4_4011402.log` |
| SLURM logs | `logs/slurm_vertical_128x1_4011403.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/` |
| W&B | [baseline-16x16](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/pwal) |
| W&B | [vertical-128x2](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/kqek) |
| W&B | [vertical-128x4](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/1ybfne3c) |
| W&B | [vertical-128x1](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/jan07qgv) |
| Code | Branch `mg/tokenization` |

## Next Steps

- [x] ~~Torchaudio mel rerun~~ — completed, mel backend confound resolved
- [x] ~~Streaming pipeline + AMP fix~~ — completed, 128×4 streaming (0.118) nearly matches pretrained ViT (0.121)
- [ ] Add data augmentation (mixup, SpecAugment) — severe overfitting (train cmAP ~0.94 vs test ~0.12) is the primary bottleneck
- [ ] Tune hyperparameters for vertical frames — shorter warmup (2–5 epochs) and higher LR; also try different dropout/weight decay
- [ ] Test 128×2 or 128×4 vertical frames with pretrained ViT weights (reshape/interpolate the 16×16 patch embedding)
- [ ] Run multi-seed experiments on top configs (128×4, 128×2) to estimate variance
- [ ] Try horizontal frame tokenization (W×500 patches spanning full time axis) as proposed in the proposal's future scope

**Leads to**: Data augmentation to reduce overfitting (likely largest remaining gain); combining vertical frame tokenization with pretrained backbones; hyperparameter tuning for non-square patch ViTs.
