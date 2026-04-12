# Experiment 027: Model Training Fixes — Results

## Context

Deep research (2026-04-11) identified two fundamental implementation errors
in our SED pipeline compared to BirdCLEF 2024-2025 winners:

1. **MixUp**: We used interpolative mixing (lambda*x1 + (1-lambda)*x2). Winners
   use additive waveform mixing (x1 + x2) with max labels. Impact: -0.036
   private LB without correct MixUp (BirdCLEF 2025 2nd place ablation).

2. **Loss**: We used BCE across all 234 species. Winners use CE on primary label
   only during training, sigmoid at inference. Impact: +0.044 private LB from
   this single change (BirdCLEF 2024 1st place).

See: `notes/2026-04-11-deep-research-claude.md`, `notes/2026-04-11-deep-research-gemini.md`

## Phase 1: v10a (Additive MixUp + CE loss)

**Config**: `configs/v10a_addmix_ce.yaml`
**Changes from v5j**: `mixup_mode=additive`, `raw_waveform_mixup=true`, `loss_type=ce`
**Everything else**: identical to v5j (5s window, LSEHead, batch 64, 20 epochs)

**SLURM**: Job 6439641, RTX 6000, 38m51s walltime
**Checkpoint**: epoch 11 (best by val_sc/auroc)

### Metrics

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.835 |
| val/cmap | 0.571 |
| val/f1_macro | 0.497 |
| val/top1_acc | 0.733 |
| val/top5_acc | 0.859 |
| val/loss | 1.597 |
| **val_sc/auroc (soundscape)** | **0.863** |
| LB | — (pending submission) |

### val_sc/auroc progression

Best checkpoint picked at epoch 11 (val_sc/auroc=0.863). Progression:

| Epoch | val_sc/auroc | Note |
|-------|-------------|------|
| 0 | 0.782 | Initial |
| 1 | 0.826 | +0.044 |
| 2 | 0.857 | +0.031 |
| 3–10 | <0.857 | Plateau (MixUp warmup ends at epoch 5) |
| **11** | **0.863** | **Best — checkpoint saved** |
| 12–19 | <0.863 | No improvement |

### Comparison with baselines

| Model | Loss | MixUp | val/auroc | val/cmap | val/top1 | val_sc/auroc | LB |
|-------|------|-------|-----------|----------|----------|-------------|-----|
| v5j (exp 019) | BCE | interp spec | 0.874 | — | — | — | **0.903** |
| v5j-dualval (exp 026) | BCE | interp spec | 0.849 | 0.596 | 0.752 | 0.956* | 0.897 |
| **v10a (exp 027)** | **CE** | **additive wfm** | **0.835** | **0.571** | **0.733** | **0.863** | **pending** |

*v5j-dualval val_sc/auroc=0.956 uses per-class filtered metric (only species with positives).
v10a val_sc/auroc=0.863 uses the same per-class filtered metric.

### Analysis

**Focal metrics (val/auroc) decreased**: 0.835 vs 0.849 (v5j-dualval) and 0.874
(v5j). This is expected with CE loss — CE forces competition between classes via
softmax, producing a different probability calibration than BCE. The sigmoid-based
val metrics (AUROC, cmap) measure sigmoid(logits), but CE optimizes softmax(logits).
The logit magnitudes are calibrated differently.

**val_sc/auroc dropped from 0.956 to 0.863**: This metric uses per-class AUROC
filtered to species with positives. The drop could indicate the CE+additive MixUp
combination is less calibrated for multi-label soundscape prediction, OR it could
be that the v5j-dualval metric was inflated by the small 86-segment val set. Only
the LB score can tell us whether this is a real regression or a calibration artifact.

**Key reminder**: Offline metrics have been anti-correlated with LB across 20+
experiments. The only reliable signal is the Kaggle LB score. Must submit to LB
before drawing conclusions.
