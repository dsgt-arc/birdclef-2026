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

**We should still submit this to LB** — the val/auroc gap might not translate
to LB (as we've seen many times). And we need the LB datapoint regardless.

## Phase 3: AttHead + Window Length Ablation

*(Training in progress — v9a/v9b/v9c submitted)*
