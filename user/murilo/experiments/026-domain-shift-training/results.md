# Experiment 026: Domain-Shift Training — Results

## Phase 2: v5j-dualval Baseline

**Config**: `configs/v5j_dualval.yaml`
**SLURM**: Job 6425604, RTX 6000, 39m55s walltime
**Checkpoint**: epoch 17 (best by val_sc/auroc)

### Metrics

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.849 |
| val/cmap | 0.596 |
| val/f1_macro | 0.504 |
| val/top1_acc | 0.752 |
| val/top5_acc | 0.866 |
| val/loss | 0.011 |
| **val_sc/auroc (soundscape)** | **0.094** |
| LB | — (pending submission) |

### Key Finding: val_sc/auroc metric is broken

The soundscape AUC of 0.094 is **not a meaningful metric**. The issue:

- Soundscape val set: 86 segments from 8 files
- Only ~30-40 of 234 species appear in those 86 segments
- `torchmetrics.AUROC(task="multilabel", average="macro")` computes per-species
  AUROC and averages across **all 234 species**
- The 194+ species with zero positives in soundscape val contribute ~0 each
- This drags the macro average down to ~0.09 regardless of model quality

The metric barely moved across training (0.046 at epoch 0 → 0.094 at epoch 17),
with increments of 0.000–0.001 per epoch. It's essentially noise and cannot
reliably guide checkpoint selection.

### Fix needed

For soundscape AUC to be useful, we need to compute it **only for species that
have ≥1 positive example** in the soundscape val set. This requires modifying
the metric computation to filter species or using a custom metric that ignores
zero-positive classes.

### Comparison with original v5j

| Metric | v5j (exp 019) | v5j-dualval (exp 026) |
|--------|---------------|----------------------|
| val/auroc | 0.874 | 0.849 |
| Best epoch | 19 | 17 (by val_sc/auroc) |
| LB | 0.903–0.906 | — (pending) |

The val/auroc difference (0.874 vs 0.849) could be due to:
1. Different checkpoint selection (epoch 17 by val_sc vs epoch 19 by val)
2. Random seed / training variance
3. The fact that val_metrics are logged at epoch 19 (end of training), not at
   the best checkpoint epoch

**LB result: 0.897** — same as the first v5j-dualval run. The soundscape-based
checkpoint selection (epoch 17) did not improve over the original v5j (0.903,
epoch 19). The checkpoint picked 2 epochs earlier but scored the same on LB.
This suggests the soundscape val metric is not selecting a better checkpoint,
though the training is identical.

## Phase 3: AttHead + Window Length Ablation

Note: Phase 3 jobs ran with the **old broken** val_sc/auroc metric (all show ~0.093).
Checkpoint selection was based on broken metric — best checkpoints may not be optimal.
The val/auroc (focal) and other metrics are still valid. Compare using focal metrics
and LB scores.

### v9a: 10s + AttHead

**Config**: `configs/v9a_10s_att.yaml`
**SLURM**: Job 6426103, RTX 6000, 1h07m walltime
**Checkpoint**: epoch 9 (best by broken val_sc/auroc)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.859 |
| val/cmap | 0.645 |
| val/f1_macro | 0.585 |
| val/top1_acc | 0.827 |
| val/top5_acc | 0.918 |
| val/loss | 0.011 |
| val_sc/auroc | 0.094 (broken) |
| LB | — (pending) |

**Comparison with baselines (focal val/auroc)**:

| Model | Window | Head | val/auroc | val/cmap | val/top1 | LB |
|-------|--------|------|-----------|----------|----------|-----|
| v5j-dualval | 5s | LSEHead | 0.849 | 0.596 | 0.752 | — |
| **v9a** | **10s** | **AttHead** | **0.859** | **0.645** | **0.827** | **—** |
| v5j (exp 019) | 5s | LSEHead | 0.874 | — | — | 0.903–0.906 |

v9a shows **+0.010 val/auroc, +0.049 cmap, +0.075 top1** over v5j-dualval.
Strong offline improvement — but offline metrics have been unreliable.
Needs LB submission to confirm.

### v9b: 15s + AttHead

**Config**: `configs/v9b_15s_att.yaml`
**SLURM**: Job 6426104, RTX 6000, 1h38m walltime
**Checkpoint**: epoch 10 (best by broken val_sc/auroc)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.858 |
| val/cmap | 0.664 |
| val/f1_macro | 0.588 |
| val/top1_acc | 0.847 |
| val/top5_acc | 0.938 |
| val/loss | 0.010 |
| val_sc/auroc | 0.094 (broken) |
| LB | — (pending) |

### Comparison across all Phase 3 models (focal metrics)

| Model | Window | Head | val/auroc | val/cmap | val/top1 | val/top5 | LB |
|-------|--------|------|-----------|----------|----------|----------|-----|
| v5j-dualval | 5s | LSEHead | 0.849 | 0.596 | 0.752 | 0.866 | 0.897 |
| v9a | 10s | AttHead | 0.859 | 0.645 | 0.827 | 0.918 | — |
| **v9b** | **15s** | **AttHead** | **0.858** | **0.664** | **0.847** | **0.938** | **—** |
| v9c | 20s | AttHead | — | — | — | — | — |

**Trend**: Longer windows show strong improvements in cmap (+0.068), top1 (+0.095),
and top5 (+0.072) vs baseline. val/auroc is similar across all (~0.858-0.859).
v9b (15s) has the best cmap and top1 so far. But offline metrics have been
unreliable — LB submission needed.

### v9c: 20s + AttHead

**Config**: `configs/v9c_20s_att.yaml`
**SLURM**: Job 6426105, RTX 6000, 2h09m walltime
**Checkpoint**: epoch 10 (best by broken val_sc/auroc)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.861 |
| val/cmap | 0.681 |
| val/f1_macro | 0.605 |
| val/top1_acc | 0.862 |
| val/top5_acc | 0.942 |
| val/loss | 0.010 |
| val_sc/auroc | 0.091 (broken) |
| LB | — (pending) |

### Full Phase 3 comparison (focal metrics, all at epoch 19 / last checkpoint)

| Model | Window | Head | val/auroc | val/cmap | val/top1 | val/top5 | LB |
|-------|--------|------|-----------|----------|----------|----------|-----|
| v5j-dualval | 5s | LSEHead | 0.849 | 0.596 | 0.752 | 0.866 | 0.897 |
| v9a | 10s | AttHead | 0.859 | 0.645 | 0.827 | 0.918 | — |
| v9b | 15s | AttHead | 0.858 | 0.664 | 0.847 | 0.938 | — |
| **v9c** | **20s** | **AttHead** | **0.861** | **0.681** | **0.862** | **0.942** | **—** |

**Trend**: Monotonic improvement across all metrics with longer windows. v9c (20s)
is the strongest model offline with +0.012 auroc, +0.085 cmap, +0.110 top1, +0.076
top5 over the 5s baseline. The cmap and top1 improvements are particularly large.

**Caveat**: offline metrics have been anti-correlated with LB in past experiments.
The v5j-dualval 0.897 LB (epoch 17, worse than v5j 0.903) is a reminder. All three
v9 models submitted with last checkpoint (epoch 19) to LB — awaiting scores.
