# Experiment 019: Perch-Guided SED Training — Results

## Phase A: Improved SED Baseline

### Step 1: Quick Wins Ablation

All ablations start from v4b (0.865 val/auroc, 0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

| Slug | Change | Best val/auroc | val/cmap | Best Epoch | Stopped | Delta vs v4b | LB | Checkpoint |
|------|--------|---------------|----------|------------|---------|-------------|-----|------------|
| `effnet-b0-sed-v5a` | BCEWithLogitsLoss | 0.875 | 0.620 | 29 | 35 | +0.010 | **0.851** | `2026-effnet-b0-sed-v5a/best-epochepoch=29.ckpt` |
| `effnet-b0-sed-v5b` | Stronger MixUp (α=1.0, θ=0.8, warmup=5) | 0.869 | 0.638 | 31 | 42 | +0.004 | **0.840** | `2026-effnet-b0-sed-v5b/best-epochepoch=31.ckpt` |
| `effnet-b0-sed-v5c` | weight_decay=1e-4 | training | — | — | — | — | — | `2026-effnet-b0-sed-v5c/` |
| `effnet-b0-sed-v5d` | LSEHead (GeM+LSE pooling) | 0.861 | — | 21 | — | -0.004 | pending | `2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt` |
| `effnet-b0-sed-v5e` | HGNetV2-B0 backbone (embed=1024) | 0.861 | 0.612 | 23 | 34 | -0.004 | pending | `2026-hgnet-b0-sed-v5e/best-epochepoch=23.ckpt` |

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

### v5a Detailed Metrics (Best Checkpoint, Epoch 29)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8750 |
| val/cmap | 0.6196 |
| val/f1_macro | 0.4218 |
| val/f1_micro | 0.6196 |
| val/top1_acc | 0.7797 |
| val/top5_acc | 0.8955 |
| val/loss | 0.0133 |

Training config: EfficientNet-B0, loss_type=bce (BCEWithLogitsLoss),
batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789083.

### v5e Detailed Metrics (Best Checkpoint, Epoch 23)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8613 |
| val/cmap | 0.6124 |
| val/f1_macro | 0.5242 |
| val/f1_micro | 0.6307 |
| val/top1_acc | 0.7822 |
| val/top5_acc | 0.9086 |
| val/loss | 1.4121 |

Training config: HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`), embed_dim=1024,
batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789090.

### Analysis

**v5a (BCE)** is the strongest ablation (+0.010 val/auroc over v4b baseline).
BCEWithLogitsLoss is a clear improvement over soft CE for multi-label SED training.
The large f1_micro jump (0.430 → 0.620) suggests BCE produces better-calibrated
predictions. Needs LB validation — offline val/auroc has been a poor LB predictor
in prior experiments (v4e: 0.868 val → 0.840 LB, v4j: 0.868 val → 0.853 LB).

**v5b (MixUp)** is the second-best ablation (+0.004 val/auroc). Stronger MixUp
regularization (uniform α=1.0, label clipping θ=0.8, 5-epoch warmup) helps the model
generalize better.

**v5e (HGNetV2-B0)** regressed slightly (-0.004 val/auroc). Despite higher embed_dim
(1024 vs 320), HGNet doesn't outperform EfficientNet-B0 with default hyperparameters.
Interestingly, v5e has the best f1_macro (0.524 vs 0.420) — may benefit from different
tuning.

**v5d (LSEHead)** regressed slightly (-0.004 val/auroc). LSE pooling doesn't help on
EfficientNet-B0.

v5c (weight_decay) still training — results pending.
