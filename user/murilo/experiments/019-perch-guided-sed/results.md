# Experiment 019: Perch-Guided SED Training — Results

## Phase A: Improved SED Baseline

### Step 1: Quick Wins Ablation

All ablations start from v4b (0.865 val/auroc, 0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

| Slug | Change | Best val/auroc | val/cmap | Best Epoch | Stopped | Delta vs v4b | LB | Checkpoint |
|------|--------|---------------|----------|------------|---------|-------------|-----|------------|
| `effnet-b0-sed-v5a` | BCEWithLogitsLoss | training | — | — | — | — | — | `2026-effnet-b0-sed-v5a/` |
| **`effnet-b0-sed-v5b`** | **Stronger MixUp (α=1.0, θ=0.8, warmup=5)** | **0.869** | **0.638** | **31** | 42 | **+0.004** | pending | `2026-effnet-b0-sed-v5b/best-epochepoch=31.ckpt` |
| `effnet-b0-sed-v5c` | weight_decay=1e-4 | training | — | — | — | — | — | `2026-effnet-b0-sed-v5c/` |
| `effnet-b0-sed-v5d` | LSEHead (GeM+LSE pooling) | 0.861 | — | 21 | — | -0.004 | pending | `2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt` |
| `effnet-b0-sed-v5e` | HGNetV2-B0 backbone (embed=1024) | training | — | — | — | — | — | `2026-hgnet-b0-sed-v5e/` |

Checkpoint base: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/`

### v5b Detailed Metrics (Best Checkpoint, Epoch 31)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8685 |
| val/cmap | 0.6376 |
| val/f1_macro | 0.4196 |
| val/f1_micro | 0.4303 |
| val/top1_acc | 0.8041 |
| val/top5_acc | 0.9126 |
| val/loss | 1.3404 |

Training config: EfficientNet-B0, mixup_alpha=1.0, mixup_theta=0.8,
mixup_warmup_epochs=5, batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789084.

### v5d Detailed Metrics (Best Checkpoint, Epoch 21)

LSEHead (Log-Sum-Exp temporal pooling) slightly regressed vs attention head.
val/auroc 0.861 vs 0.865 baseline (-0.004).

### Analysis

**v5b (MixUp)** is the strongest ablation so far (+0.004 val/auroc). Stronger MixUp
regularization (uniform α=1.0, label clipping θ=0.8, 5-epoch warmup) helps the model
generalize better. The improvement is modest but consistent — worth combining with
other positive ablations in Step 2.

**v5d (LSEHead)** regressed slightly. LSE pooling may work better on HGNet (where
ttahara used it at 0.884 LB) than on EfficientNet-B0. Will revisit if v5e (HGNet)
shows promise.

Remaining runs (v5a, v5c, v5e) still training — results expected within ~4h.
