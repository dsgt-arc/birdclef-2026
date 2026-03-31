# Experiment 019: Perch-Guided SED Training — Results

## Phase A: Improved SED Baseline

### Step 1: Quick Wins Ablation

All ablations start from v4b (0.865 val/auroc, 0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

| Slug | Change | Best val/auroc | val/cmap | Best Epoch | Stopped | Delta vs v4b | LB | Checkpoint |
|------|--------|---------------|----------|------------|---------|-------------|-----|------------|
| `effnet-b0-sed-v5a` | BCEWithLogitsLoss | 0.875 | 0.620 | 29 | 35 | +0.010 | **0.851** | `2026-effnet-b0-sed-v5a/best-epochepoch=29.ckpt` |
| `effnet-b0-sed-v5b` | Stronger MixUp (α=1.0, θ=0.8, warmup=5) | 0.869 | 0.638 | 31 | 42 | +0.004 | **0.840** | `2026-effnet-b0-sed-v5b/best-epochepoch=31.ckpt` |
| `effnet-b0-sed-v5c` | weight_decay=1e-4 | 0.867 | 0.624 | 35 | 42 | +0.002 | pending | `2026-effnet-b0-sed-v5c/best-epochepoch=35.ckpt` |
| `effnet-b0-sed-v5d` | LSEHead (GeM+LSE pooling) | 0.861 | — | 21 | — | -0.004 | pending | `2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt` |
| `effnet-b0-sed-v5e` | HGNetV2-B0 backbone (embed=1024) | 0.861 | 0.612 | 23 | 34 | -0.004 | pending | `2026-hgnet-b0-sed-v5e/best-epochepoch=23.ckpt` |

Checkpoint base: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/`

### Step 2: ttahara Full Recipe Reproduction

Faithful reproduction of ttahara's 0.884 LB training pipeline (LSEHead with mean freq,
BCEWithLogitsLoss, per-sample MixUp α=1.0/θ=0.8, OneCycleLR, 5s segments, 20 epochs).
Two backbone variants to isolate recipe vs architecture impact.

Reference: `user/murilo/notes/ttahara-hgnetv2-b0-annotated.md`

| Slug | Backbone | Best val/auroc | val/cmap | Best Epoch | Stopped | LB | Checkpoint |
|------|----------|---------------|----------|------------|---------|-----|------------|
| `effnet-b0-sed-v5f` | EfficientNet-B0 (embed=320) | **0.866** | 0.599 | 14 | 20 | **0.855** | `2026-effnet-b0-sed-v5f/best-epochepoch=14.ckpt` |
| `hgnet-b0-sed-v5g` | HGNetV2-B0 (embed=1024) | **0.872** | 0.618 | 18 | 20 | **0.892** | `2026-hgnet-b0-sed-v5g/best-epochepoch=18.ckpt` |

Key differences from Step 1 ablations: LSEHead(freq_pool=mean), BCE loss (clipwise only),
per-sample MixUp(α=1.0, θ=0.8, warmup=5, 100%), OneCycleLR(lr=5e-4 uniform, wd=1e-4),
torchaudio mel(n_fft=2048, resize 256x256, min-max norm), 5s segments, batch=64, 20 epochs.

Expected: v5g single-fold ~0.870 (ttahara 4-fold = 0.884).

### v5g Detailed Metrics (Best Checkpoint, Epoch 18)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8720 |
| val/cmap | 0.6180 |
| val/f1_macro | 0.5191 |
| val/f1_micro | 0.6614 |
| val/top1_acc | 0.7393 |
| val/top5_acc | 0.8515 |
| val/loss | 0.0125 |

Training config: HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`), LSEHead(mean freq,
hidden=1024), BCEWithLogitsLoss, per-sample MixUp(α=1.0, θ=0.8, warmup=5),
OneCycleLR(lr=5e-4, wd=1e-4), 5s segments, 20 epochs, batch=64.
GPU: SLURM job 5953608. **Training time: ~21 min** (vs ~3-4h for v4b/v5a-e).

### v5f Detailed Metrics (Best Checkpoint, Epoch 14)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8657 |
| val/cmap | 0.5985 |
| val/f1_macro | 0.4936 |
| val/f1_micro | 0.6718 |
| val/top1_acc | 0.7379 |
| val/top5_acc | 0.8582 |
| val/loss | 0.0125 |

Training config: EfficientNet-B0, LSEHead(mean freq, hidden=320), BCEWithLogitsLoss,
per-sample MixUp(α=1.0, θ=0.8, warmup=5), OneCycleLR(lr=5e-4, wd=1e-4), 5s segments,
20 epochs, batch=64. GPU: SLURM job 5953606.

### Analysis

**v5g (0.872) matches our target.** The full ttahara recipe on HGNetV2-B0 achieves
0.872 val/auroc, within range of the ~0.870 single-fold expectation. Key comparisons:

| Model | Recipe | val/auroc | Delta vs v4b |
|-------|--------|----------|-------------|
| v4b (EfficientNet) | v4b recipe | 0.865 | baseline |
| v5e (HGNet) | v4b recipe | 0.861 | -0.004 (backbone alone hurt) |
| **v5f (EfficientNet)** | **ttahara recipe** | **0.866** | **+0.001** (recipe only) |
| **v5g (HGNet)** | **ttahara recipe** | **0.872** | **+0.007** (recipe + backbone) |

**Recipe impact** (same backbone): v5f (0.866) vs v4b (0.865) = +0.001 on EfficientNet.
The recipe alone barely moves the needle on EfficientNet — the gains are mostly from the backbone.

**Backbone impact** (same recipe): v5g (0.872) vs v5f (0.866) = +0.006 from HGNet.
HGNet is clearly the better backbone under the ttahara recipe.

**Recipe + backbone combined**: v5e (0.861, HGNet+old) → v5g (0.872, HGNet+ttahara) = +0.011.
The ttahara recipe unlocks HGNet's potential that the v4b recipe was suppressing.

**Training speed**: 21 min for 20 epochs (5s, batch=64, no grad accum) vs ~3-4h for v4b
recipe (50 epochs, 10s, 4× grad accum). The ttahara recipe is 8-10× faster to train.

**LB validation needed**: offline val/auroc has been a poor LB predictor in prior experiments.
v5g's 0.872 offline is similar to v5a's 0.875 (which scored only 0.851 LB). The question is
whether the ttahara recipe produces better-calibrated predictions that transfer to LB.

**Next**: If v5g LB ≥ 0.870 → proceed to 4-fold ensemble (target ~0.884).

### Step 3: 4-Fold Ensemble (v5g-4fold)

MultiLabelStratifiedGroupKFold (n_splits=4, seed=1086), grouped by audio_id (filename stem).
Per-segment soundscape entries for stratification. Trained with `--fold N` flag.

| Fold | val/auroc | val/cmap | Best Epoch | Checkpoint |
|------|----------|----------|------------|------------|
| 0 | **0.939** | 0.545 | 14 | `fold0/best-epochepoch=14.ckpt` |
| 1 | 0.842 | 0.566 | 19 | `fold1/best-epochepoch=19.ckpt` |
| 2 | 0.826 | 0.557 | 18 | `fold2/best-epochepoch=18.ckpt` |
| 3 | 0.802 | 0.556 | 16 | `fold3/best-epochepoch=16.ckpt` |

**Fold imbalance issue**: Fold 0 val/auroc (0.939) is 0.137 higher than fold 3 (0.802).
The root cause is soundscape file distribution: fold 0 has 55 of 66 soundscape files in
its validation set (and only 11 for training), while folds 2-3 have only 2 each in val.
With only 66 soundscape file-level groups, the GroupKFold constraint prevents even
distribution. The per-segment stratification fix (from per-file) barely changed this
(57→55 files in fold 0).

**Impact**: Fold 0 model trains on very few soundscape files, potentially weaker on
soundscape-like test data. The ensemble averaging may still help (each fold's weakness
is different), but the extreme spread suggests the folds are not comparable.

### LB Results

| Submission | LB | Notes |
|-----------|-----|-------|
| v5g single-fold | **0.892** | +0.032 over v4b (0.860), +0.037 over v5f (0.855) |
| **v5g-4fold ensemble** | **0.901** | **Broke 0.900 barrier!** +0.009 from 4-fold, +0.041 over v4b |

**v5g (0.892 LB)** is a massive improvement. The ttahara recipe + HGNet backbone
translates to LB far better than any prior SED model. Notably, offline val/auroc
(0.872) UNDER-predicted LB (0.892) — the opposite of all prior experiments where
val over-predicted LB. The ttahara recipe produces better-calibrated predictions.

**v5g-4fold (0.901 LB)** confirms the ensemble adds +0.009 despite the fold imbalance
(fold 0 has 55/66 soundscape files). This is within the expected +0.010-0.015 range
from ttahara's 4-fold approach.

Comparison to competition:
- ttahara HGNet 4-fold: 0.884 LB (our reproduction: **0.901**, +0.017 above reference)
- Our Perch ProtoSSM v4: 0.918 LB (still higher, but SED closing the gap)
- Salman single EfficientNet-B0 SED: 0.908 (we're at 0.892 single-fold, gap = 0.016)

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

**v5c (weight_decay=1e-4)** had marginal improvement (+0.002 val/auroc over v4b). Lower
weight decay doesn't meaningfully change performance — the model is not significantly
over-regularized at the default 0.01.

### v5c Detailed Metrics (Best Checkpoint, Epoch 35)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8669 |
| val/cmap | 0.6240 |
| val/f1_macro | 0.3987 |
| val/f1_micro | 0.3865 |
| val/top1_acc | 0.8027 |
| val/top5_acc | 0.9114 |
| val/loss | 1.3720 |

Training config: EfficientNet-B0, weight_decay=1e-4 (default 0.01),
batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789087.
