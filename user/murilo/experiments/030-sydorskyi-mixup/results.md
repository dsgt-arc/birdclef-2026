---
experiment: "030-sydorskyi-mixup"
status: in-progress
date: 2026-04-26
author: Murilo Gustineli
---

# Experiment 030: Sydorskyi-faithful MixUp on v5j — Results

## Status

**v1 (with `dual_validation: true`)**: training + LB COMPLETE.
**v2 (no `dual_validation`)**: training COMPLETE for all 4 cells (2026-04-26
overnight). LB pending — ONNX export + Kaggle push + submission next.

### v1 LB results (all 4 cells)

| | control (spec, interp) | sydorskyi (wave, max) | mixup Δ |
|---|---:|---:|---:|
| **HGNetV2-B0** (v5j) | 0.890 | 0.892 | +0.002 (null) |
| **EfficientNetV2-B0** | 0.893 | **0.901** | **+0.008** (small positive) |
| backbone Δ | +0.003 | +0.009 | |

**Best v1 cell**: `effnetv2b0_sydorskyi` at LB 0.901 — the diagonal corner.

### Why all 4 v1 cells are below v5j 0.906

`dual_validation: true` (added between exp 019 and exp 030) changed `val/auroc`
from "macro-AUROC over the COMBINED focal+SS val" to "focal-only" via the
new `val_dataloader()` returning `[focal_loader, sc_loader]` instead of one
ConcatDataset loader. Because the model_checkpoint callback monitors
`val/auroc`, this changed which checkpoint gets saved as best (focal-only
val peaks at a different epoch than combined val).

The cost was documented in exp 026: v5j-dualval = 0.897 LB vs v5j = 0.906 LB
(−0.009). Our v1 runs landed at 0.890-0.901, consistent with that regime
(dual-validation cost compounded with the architecture/mixup variants).

### v2 plan: retrain all 4 with dual_validation removed

To get a clean comparison vs v5j 0.906, all 4 variants are being retrained
with `dual_validation: true` removed. Run names get `-v2` suffix to keep
v1 checkpoints intact for forensic comparison.

Expected v2 outcomes (assuming the −0.009 dual_val cost is recoverable):
- v5j_control_v2: ~0.899-0.906 (reproduces v5j; pipeline reproducibility check)
- v5j_sydorskyi_v2: ~0.899-0.906 (mixup change null on HGNet, per v1 result)
- effnetv2b0_control_v2: ~0.900-0.905 (backbone null on its own, per v1)
- effnetv2b0_sydorskyi_v2: **~0.908-0.912** (best v1 cell + dual_val cost recovery)

If effnetv2b0_sydorskyi_v2 lands ≥ 0.908, it would be **the first single-model
improvement over v5j 0.906** for us in this competition. That's the
load-bearing test.

## Context

Tests whether replacing v5j's Zhang-style interpolated spectrogram MixUp with
Sydorskyi 2025 2nd-place's faithful waveform recipe — `wave = (w1+w2)/2 +
peak-normalize`, `target = max(y1, y2)` — moves LB. Single-variable change
on v5j (Phase A). See [proposal.md](proposal.md) for full hypothesis,
method, and decision matrix.

Motivated by three findings from PR #42 (D1 diagnosis):
- MixUp implementation audit: v5j uses interpolated targets, Sydorskyi (LB
  0.917 single fold, top-3 finishers all do this) uses `max(y1, y2)`.
- Target-distribution simulation: 80% of v5j's active training target values
  are sub-1, 59% of training samples have NO class at full presence.
- D1 verdict: 84% top-1% failure overlap between v5j and v6k. Failure
  pattern is data-driven; target-distribution mismatch is the most concrete
  actionable lever.

## Phase A: Training — DONE

### Setup confirmation

Both runs confirmed in SLURM logs:
- `INFO:__main__:GPU mel + waveform mixup: mixup before mel on GPU` →
  `waveform_mixup=True` correctly read.
- v5j_sydorskyi config dump shows `mixup_mode: "average_max"`, `mixup_prob: 0.5`,
  `mixup_warmup_epochs: 0`, `mixup_theta: 1.0` — all four planned changes present.
- No errors, NaNs, or stability issues during training. Both jobs exit 0.
- Best-by-val checkpoints saved at epoch 14 for BOTH variants.

### v5j_control (reproducibility baseline)

| Metric | Value |
|---|---:|
| **val/auroc** | **0.8490** |
| val/cmap | 0.5959 |
| val/f1_macro | 0.5039 |
| val/f1_micro | 0.6855 |
| val/top1_acc | 0.7516 |
| val/top5_acc | 0.8665 |
| val/loss | 0.0114 |
| val_sc/auroc * | 0.9547 |
| val_sc/cmap * | 0.8532 |
| train/loss_epoch (final) | 0.0156 |
| **best_val_score** | **0.8516** (epoch 14) |
| Walltime | 37m 44s on rtx6000 |
| SLURM job ID | 7452610 |
| Checkpoint | `/storage/project/.../experiments/030-sydorskyi-mixup/checkpoints/2026-exp030-v5j-control/best-epochepoch=14.ckpt` |

`* val_sc/*` is the broken metric per exp 026 caveat (macro-AUROC over 234
species when only ~35 appear in 86 SS val segments).

**Reproducibility check**: v5j_control val/auroc 0.8490 vs v5j-control
(exp 027) val/auroc 0.851 — Δ −0.002, well within Mahog's reported seed
noise (±0.005). **Pipeline reproducibility confirmed.** No drift in our
codebase since exp 027.

### v5j_sydorskyi (the test)

| Metric | Value | Δ vs control |
|---|---:|---:|
| **val/auroc** | **0.8472** | **−0.002** |
| val/cmap | 0.6065 | **+0.011** |
| val/f1_macro | 0.5467 | **+0.043** |
| val/f1_micro | 0.6948 | +0.009 |
| val/top1_acc | 0.7516 | +0.000 |
| val/top5_acc | 0.8613 | −0.005 |
| val/loss | 0.0118 | +0.0004 |
| val_sc/auroc * | 0.9660 | +0.011 |
| val_sc/cmap * | 0.8774 | +0.024 |
| train/loss_epoch (final) | 0.0178 | **+0.0022** |
| **best_val_score** | **0.8507** (epoch 14) | −0.001 |
| Walltime | 41m 45s on rtx6000 | +4 min (peak-norm overhead) |
| SLURM job ID | 7452611 | |
| Checkpoint | `/storage/project/.../experiments/030-sydorskyi-mixup/checkpoints/2026-exp030-v5j-sydorskyi/best-epochepoch=14.ckpt` |

### Side-by-side summary

| Metric | v5j_control | v5j_sydorskyi | Δ | Interpretation |
|---|---:|---:|---:|---|
| val/auroc | 0.8490 | 0.8472 | **−0.002** | Within seed noise — flat |
| val/cmap | 0.5959 | 0.6065 | **+0.011** | Notable improvement (precision-based) |
| val/f1_macro | 0.5039 | 0.5467 | **+0.043** | Big improvement (threshold-based) |
| val/loss | 0.0114 | 0.0118 | +0.0004 | Slightly higher (harder targets) |
| train/loss_epoch | 0.0156 | 0.0178 | **+0.0022** | Higher train loss — model fitting harder |

### Convergence shape

Both runs landed at epoch 14 as the best-by-val checkpoint, identical
trajectory shape early. Sydorskyi's `train/loss_epoch` is consistently
higher across epochs — expected because `max(y1, y2)` targets are harder
to fit than interpolated `lam·y1 + (1-lam)·y2` (the model has to push
BOTH classes to 1.0 in mixed samples rather than splitting weight).

This is potentially a **good** sign for generalization: less overfitting
to training-set memorization, more pressure to learn shared features.
But it's a hypothesis — only LB will tell.

## Interpretation so far (offline only — LB pending)

The offline picture is mixed but internally consistent:

- **AUROC** (rank-based): essentially flat. Both variants rank species
  about equally well on val. The training change didn't disturb relative
  ordering of predictions.
- **cmap and f1_macro** (precision/threshold-based): meaningfully higher
  for sydorskyi. cmap +0.011, f1_macro +0.043. These metrics reward
  per-species precision more than AUROC does.
- **train loss higher**: fits the prediction. Sum-clip targets are
  structurally harder; the model can't trivially memorize.
- **val_sc metrics higher**: directionally consistent with cmap/f1_macro
  improvement, but the metric is broken so we shouldn't read too much into it.

Why might cmap improve while AUROC stays flat?
- AUROC is invariant to the calibration of probabilities (only the
  ordering matters).
- cmap is precision-at-rank averaged over species; it cares about the
  *level* of confidence, not just ordering.
- Sydorskyi's `max(y1, y2)` mixup pushes the model to predict BOTH species
  at 1.0 in mixed examples → training pressure for higher-confidence
  positive predictions → potentially better-calibrated outputs on
  multi-label test cases (which is what cmap measures).

This is consistent with the hypothesis that target-distribution mismatch
matters: training the model on a target distribution closer to the test
distribution improves precision-based metrics that match the test scoring
structure (LB is also macro-precision-based).

**But**: offline metrics have been **anti-correlated with LB** historically.
Cases where offline cmap/f1 improved but LB regressed are common. The cmap
improvement is encouraging but NOT a confident LB prediction.

## Phase A: Training trajectory (W&B)

Both runs available at W&B project `birdclef-2026-exp030`:
- `2026-exp030-v5j-control` (run id `el93xunm`)
- `2026-exp030-v5j-sydorskyi` (run id `3bjn6i3y`)

Use the W&B UI to compare:
- `train/loss_epoch` — should diverge mid-training (sydorskyi higher)
- `val/cmap` — should diverge favoring sydorskyi
- `val/auroc` — should track each other closely
- Both should peak around epoch 14

## Phase B: EfficientNetV2-B0 2x2 — DONE

### Setup confirmation

- `INFO:birdclef.encoders.efficientnet:EfficientNetEncoder: tf_efficientnetv2_b0.in1k, embed_dim=192, stages=5`
  — backbone loaded correctly. **Note**: actual feature dim is **192 channels**
  (stage-5 of EfficientNetV2-B0), NOT the 1280 we set in the config. The
  encoder takes the spatial feature map at stage 5, ignoring the config's
  `embed_dim` value. The 1280 figure is the post-pool projection in standard
  classifier-head usage; we use the spatial features. Both EffNetV2-B0 cells
  share this same effective architecture so the within-backbone mixup
  comparison is internally clean — but the v2-B0-vs-HGNet comparison is
  at very different feature dims (192 vs 1024).
- Total params: 6.0M (vs HGNet's 3.1M). LSEHead's input projection is now
  Linear(192 → 1024), so the head is smaller (437K params) and the backbone
  is bigger.
- Both runs confirmed log lines: control `GPU mel + spectrogram mixup`,
  sydorskyi `GPU mel + waveform mixup: mixup before mel on GPU`.
- No errors / NaN / instability. Both jobs exit 0. Best epoch = 19 for both
  (later than HGNet's 14 — bigger backbone keeps improving longer).

### effnetv2b0_control (v2-B0 + spec/interpolated mixup)

| Metric | Value | Δ vs HGNet control |
|---|---:|---:|
| **val/auroc** | **0.8510** | +0.002 |
| val/cmap | 0.6124 | +0.017 |
| val/f1_macro | 0.5320 | +0.028 |
| val/loss | 0.0109 | −0.0005 |
| val_sc/auroc * | 0.9278 | −0.027 |
| val_sc/cmap * | 0.8163 | −0.037 |
| train/loss_epoch | 0.0146 | −0.001 |
| **best_val_score** | **0.8510** (epoch 19) | — |
| Walltime | 41m 17s | +3.5 min |
| SLURM job ID | 7454128 | — |
| W&B run id | `jhfx75pb` | — |

### effnetv2b0_sydorskyi (v2-B0 + Sydorskyi mixup) — best cell

| Metric | Value | Δ vs HGNet sydorskyi | Δ vs v2-B0 control |
|---|---:|---:|---:|
| **val/auroc** | **0.8518** | +0.005 | +0.001 |
| val/cmap | **0.6303** | +0.024 | +0.018 |
| val/f1_macro | **0.5751** | +0.028 | +0.043 |
| val/loss | 0.0109 | −0.0009 | +0.000 |
| val_sc/auroc * | 0.9416 | −0.024 | +0.014 |
| val_sc/cmap * | 0.8535 | −0.024 | +0.037 |
| train/loss_epoch | 0.0166 | −0.001 | +0.002 |
| **best_val_score** | **0.8518** (epoch 19) | — | — |
| Walltime | 45m 37s | +3.9 min | +4.4 min |
| SLURM job ID | 7454131 | — | — |
| W&B run id | `bampbee6` | — | — |

### 2x2 offline summary

Reading: `val/auroc / val/cmap / val/f1_macro`

| | control (spec, interp) | sydorskyi (wave, max) |
|---|---|---|
| **HGNetV2-B0** (v5j) | 0.8490 / 0.5959 / 0.5039 | 0.8472 / 0.6065 / 0.5467 |
| **EfficientNetV2-B0** | 0.8510 / 0.6124 / 0.5320 | **0.8518 / 0.6303 / 0.5751** |

The bottom-right cell (EfficientNetV2-B0 + Sydorskyi mixup) is the best on
every offline metric except val/auroc, where it ties the v2-B0 control.

### Main effects (averaging across the other axis)

**Backbone main effect** (EffNetV2-B0 − HGNetV2-B0, averaged over mixup):

| Metric | Δ |
|---|---:|
| val/auroc | **+0.003** |
| val/cmap | **+0.020** |
| val/f1_macro | **+0.028** |
| val/loss | −0.001 |

**MixUp main effect** (sydorskyi − control, averaged over backbone):

| Metric | Δ |
|---|---:|
| val/auroc | **−0.001** (flat) |
| val/cmap | **+0.014** |
| val/f1_macro | **+0.043** |
| val/loss | +0.000 |

### Interaction term

`(v2-B0 sydorskyi − v2-B0 control) − (HGNet sydorskyi − HGNet control)` per metric:

| Metric | Interaction Δ | Interpretation |
|---|---:|---|
| val/auroc | +0.003 | Sydorskyi mixup helps slightly more on v2-B0 than HGNet |
| val/cmap | +0.007 | Sydorskyi mixup helps more on v2-B0 |
| val/f1_macro | +0.000 | Same magnitude on both backbones |

Mostly additive: the two improvements (backbone + mixup) stack roughly
independently on cmap and f1_macro. Slight positive interaction on auroc
and cmap (sydorskyi is *slightly* more helpful on the bigger backbone).

### Phase B observations

1. **EfficientNetV2-B0 is offline-better than HGNetV2-B0** across both mixup
   conditions, even at our atypical 192-channel feature depth. cmap +0.020,
   f1_macro +0.028 averaged across mixup. Salman's backbone choice has
   empirical support in our pipeline.

2. **Sydorskyi mixup helps offline on BOTH backbones**, especially on cmap
   (+0.014) and f1_macro (+0.043). Mixup main effect is consistent across
   architectures.

3. **AUROC is essentially flat across the entire 2x2** (range 0.847-0.852).
   The signal is not in rank-ordering; it's in the *calibration* of
   confidences — exactly what cmap and f1_macro measure.

4. **The "best" cell offline is `effnetv2b0_sydorskyi`**: val/auroc 0.8518,
   val/cmap 0.6303, val/f1_macro 0.5751. Compared to v5j-original
   (HGNetV2-B0 + spec mixup, our 0.906 LB), the offline improvements are:
   - val/auroc: +0.003 (negligible)
   - val/cmap: +0.034 (huge)
   - val/f1_macro: +0.071 (huge)

5. **Train loss pattern matches the structural prediction**: Sydorskyi
   mixup → harder targets → higher train loss. EffNetV2-B0 → more capacity
   → lower train loss. Both effects compose: v2-B0 sydorskyi has the lowest
   val/loss (0.0109, tied with v2-B0 control) and intermediate train loss
   (0.0166).

### Caveats specific to Phase B

- **Feature-dim mismatch**: our config sets `embed_dim: 1280` for v2-B0 but
  the encoder uses 192-channel stage-5 features. The two v2-B0 cells share
  this same effective architecture so within-backbone comparison is clean,
  but a "faithful" v2-B0 setup might use a different feature extraction
  layer with more channels. Worth investigating if Phase B LB confirms.
- **Best epoch shifted from 14 (HGNet) to 19 (v2-B0)**: bigger backbone
  keeps improving longer, but we cap at 20 epochs. v2-B0 might benefit from
  more epochs — it didn't plateau cleanly.

## Phase B trajectory (W&B)

- `2026-exp030-effnetv2b0-control` (run id `jhfx75pb`)
- `2026-exp030-effnetv2b0-sydorskyi` (run id `bampbee6`)

## Combined offline picture (all 4 cells, v1)

The Sydorskyi mixup recipe and EfficientNetV2-B0 backbone both improve our
offline cmap/f1_macro metrics, and they stack roughly additively. The best
cell (`effnetv2b0_sydorskyi`) hits val/cmap 0.6303 — a +0.034 improvement
over v5j (0.5959) on cmap.

But this is offline. The historical offline-LB anti-correlation is the
strong prior (val/auroc 0.874 → 0.906 LB for v5j, vs val/auroc 0.874 →
0.899 LB for v6l manifold mixup). cmap improvements have not consistently
translated to LB gains.

## Phase A + Phase B v2: Training — DONE

All 4 variants retrained with `dual_validation: true` removed; run names
get `-v2` suffix so v1 checkpoints stay intact. Same seed, same recipe
otherwise. Trained overnight 2026-04-26 → 2026-04-27, all 4 SLURM jobs
exited 0, all checkpoints saved.

### v5j_control_v2 (HGNet + spec interp; reproducibility baseline)

| Metric | Value | Δ vs v5j_control v1 |
|---|---:|---:|
| **val/auroc** | **0.8728** | **+0.024** |
| val/cmap | 0.6268 | +0.031 |
| val/f1_macro | 0.5270 | +0.023 |
| val/f1_micro | 0.6844 | −0.001 |
| val/top1_acc | 0.7567 | +0.005 |
| val/top5_acc | 0.8694 | +0.003 |
| val/loss | 0.0120 | +0.0006 |
| **best_val_score** | **0.8747** (epoch 15) | +0.023 |
| Walltime | 40m 12s | +2m 28s |
| SLURM job ID | 7455688 | — |
| W&B run id | `586p423f` | — |
| Checkpoint | `2026-exp030-v5j-control-v2/best-epochepoch=15.ckpt` | — |

### v5j_sydorskyi_v2 (HGNet + Sydorskyi mixup)

| Metric | Value | Δ vs v5j_sydorskyi v1 |
|---|---:|---:|
| **val/auroc** | **0.8700** | **+0.023** |
| val/cmap | 0.6339 | +0.027 |
| val/f1_macro | 0.5663 | +0.020 |
| val/f1_micro | 0.6984 | +0.004 |
| val/top1_acc | 0.7567 | +0.005 |
| val/top5_acc | 0.8646 | +0.003 |
| val/loss | 0.0123 | +0.0005 |
| **best_val_score** | **0.8737** (epoch 14) | +0.023 |
| Walltime | 1h 14m 54s | +33m (queue/node variance) |
| SLURM job ID | 7455689 | — |
| W&B run id | `e6hdxkg8` | — |
| Checkpoint | `2026-exp030-v5j-sydorskyi-v2/best-epochepoch=14.ckpt` | — |

### effnetv2b0_control_v2 (EffNetV2-B0 + spec interp)

| Metric | Value | Δ vs effnetv2b0_control v1 |
|---|---:|---:|
| **val/auroc** | **0.8745** | **+0.024** |
| val/cmap | 0.6461 | +0.034 |
| val/f1_macro | 0.5545 | +0.022 |
| val/f1_micro | 0.7120 | — |
| val/top1_acc | 0.7777 | — |
| val/top5_acc | 0.8758 | — |
| val/loss | 0.0117 | +0.0008 |
| **best_val_score** | **0.8745** (epoch 19) | +0.024 |
| Walltime | 41m 5s | −0m 12s |
| SLURM job ID | 7455690 | — |
| W&B run id | `fyqjktl5` | — |
| Checkpoint | `2026-exp030-effnetv2b0-control-v2/best-epochepoch=19.ckpt` | — |

### effnetv2b0_sydorskyi_v2 (EffNetV2-B0 + Sydorskyi mixup) — best cell

| Metric | Value | Δ vs effnetv2b0_sydorskyi v1 |
|---|---:|---:|
| **val/auroc** | **0.8760** | **+0.024** |
| val/cmap | **0.6556** | +0.025 |
| val/f1_macro | **0.5922** | +0.017 |
| val/f1_micro | 0.7241 | — |
| val/top1_acc | 0.7870 | — |
| val/top5_acc | 0.8803 | — |
| val/loss | 0.0116 | +0.0007 |
| **best_val_score** | **0.8760** (epoch 19) | +0.024 |
| Walltime | 41m 16s | −4m 21s |
| SLURM job ID | 7455691 | — |
| W&B run id | `cg9v7ksj` | — |
| Checkpoint | `2026-exp030-effnetv2b0-sydorskyi-v2/best-epochepoch=19.ckpt` | — |

### v2 reproducibility check vs v5j-original

v5j-original from exp 019 had val/auroc ≈ 0.874 → LB 0.906. With dual_val
removed in v2, val/auroc is the combined focal+SS macro-AUROC again, which
is directly comparable to v5j-original.

| | val/auroc | Δ vs v5j 0.874 | Within seed noise (±0.005)? |
|---|---:|---:|---|
| v5j_control_v2 | **0.8728** | −0.001 | ✓ YES |
| v5j_sydorskyi_v2 | 0.8700 | −0.004 | ✓ YES |
| effnetv2b0_control_v2 | 0.8745 | +0.001 | ✓ YES |
| effnetv2b0_sydorskyi_v2 | **0.8760** | **+0.002** | ✓ YES |

**Pipeline reproducibility CONFIRMED in the v2 regime** — v5j_control_v2 lands
−0.001 from the v5j-original 0.874. The dual_validation regime cost (~+0.024
val/auroc when removed) matches the exp 026 documentation.

### v2 2x2 offline summary

Reading: `val/auroc / val/cmap / val/f1_macro`

| | control (spec, interp) | sydorskyi (wave, max) |
|---|---|---|
| **HGNetV2-B0** (v5j arch) | 0.8728 / 0.6268 / 0.5270 | 0.8700 / 0.6339 / 0.5663 |
| **EfficientNetV2-B0** | 0.8745 / 0.6461 / 0.5545 | **0.8760 / 0.6556 / 0.5922** |

Best v2 cell on every offline metric: `effnetv2b0_sydorskyi_v2` —
val/auroc 0.8760, val/cmap 0.6556, val/f1_macro 0.5922.

### v2 main effects

**Backbone main effect** (EffNetV2-B0 − HGNetV2-B0, averaged over mixup):

| Metric | v2 Δ | v1 Δ | Match? |
|---|---:|---:|---|
| val/auroc | +0.004 | +0.003 | ✓ |
| val/cmap | +0.020 | +0.020 | ✓ |
| val/f1_macro | +0.027 | +0.028 | ✓ |

**MixUp main effect** (sydorskyi − control, averaged over backbone):

| Metric | v2 Δ | v1 Δ | Match? |
|---|---:|---:|---|
| val/auroc | −0.001 | −0.001 | ✓ |
| val/cmap | +0.008 | +0.014 | smaller in v2 |
| val/f1_macro | +0.039 | +0.043 | similar |

**Interaction** ((v2-B0 sydorskyi − v2-B0 control) − (HGNet sydorskyi − HGNet control)):

| Metric | v2 Δ | v1 Δ |
|---|---:|---:|
| val/auroc | +0.004 | +0.003 |
| val/cmap | +0.002 | +0.007 |
| val/f1_macro | −0.002 | +0.000 |

Conclusion: **v2 offline structure replicates v1**. Backbone main effect is
robust (+0.004 auroc / +0.020 cmap / +0.027 f1_macro across both regimes).
Mixup main effect on cmap is smaller in v2 but still positive (+0.008 vs
+0.014). Mixup main effect on f1_macro and the slight positive interaction
on auroc both reproduce.

### v2 vs v1 — what changed

The +0.024 val/auroc shift across all 4 cells from v1 → v2 is consistent with
the dual_validation hypothesis: when `dual_validation: true`, val/auroc is
focal-only (~0.85 range); when removed, it's macro-AUROC over the combined
focal+SS val (~0.87 range, since SS samples push macro-AUROC up). All 4 v2
cells shifted by ~+0.024, ruling out any cell-specific drift.

The 2x2 *structure* (backbone +, mixup +, additive interaction) reproduced
across v1 and v2. This means the LB will likely show the same ordering as
v1 (`effnetv2b0_sydorskyi` > `effnetv2b0_control` > `v5j_sydorskyi` >
`v5j_control`) — but each cell should land ~+0.009 LB above its v1 value
if the dual_val cost (per exp 026) is fully recoverable.

### v2 LB predictions (decision matrix)

If the dual_val confound is fully responsible for the v1 LB shortfall:

| Cell | v1 LB | Predicted v2 LB | If lands in this band |
|---|---:|---:|---|
| v5j_control_v2 | 0.890 | 0.899-0.906 | dual_val confirmed; pipeline reproduces |
| v5j_sydorskyi_v2 | 0.892 | 0.899-0.906 | mixup null on HGNet (consistent w/ v1) |
| effnetv2b0_control_v2 | 0.893 | 0.901-0.905 | backbone null on control (consistent w/ v1) |
| **effnetv2b0_sydorskyi_v2** | **0.901** | **0.908-0.912** | **first single-model improvement over v5j 0.906** |

Headline question: does `effnetv2b0_sydorskyi_v2` clear ≥ 0.908?

## Combined offline picture (all 4 cells, v2)

The 2x2 offline structure from v1 holds in v2 (backbone +0.020 cmap / mixup
+0.008 cmap, mostly additive). The best cell is the diagonal corner
`effnetv2b0_sydorskyi_v2` on every metric — val/auroc 0.8760, val/cmap
0.6556, val/f1_macro 0.5922. Vs v5j-original 0.874 val/auroc, this is +0.002
auroc — within seed noise on auroc, but with materially higher cmap and
f1_macro.

Whether that translates to LB depends on whether the cmap/f1 improvements
break the historical offline-LB anti-correlation. We've seen offline cmap
improvements regress on LB before (22+ training-recipe failures). But this
time the 2x2 structure is internally consistent across v1 LBs (already in)
and v2 offline metrics, which is a stronger signal than offline-only.

## Open work

### v2 cycle (training DONE — pipeline next)

- [x] All 4 v2 SLURM jobs complete; metrics recorded above
- [x] v5j_control_v2 reproduces v5j-original 0.874 val/auroc within ±0.005
  → pipeline reproducibility confirmed for v2 regime
- [ ] ONNX export for all 4 v2 checkpoints (4 × `bash sbatch/run.sh export
  --model {hgnet-b0-sed-v5j-control, hgnet-b0-sed-v5j-sydorskyi,
  effnetv2-b0-sed-control, effnetv2-b0-sed-sydorskyi}`)
- [ ] Re-upload weights + push kernels (existing-model path; same Kaggle slugs
  as v1, replaces in place): 4 × `python scripts/upload_kaggle.py --model X
  --push-kernel`
- [ ] Submit 4 kernels from Kaggle UI; collect LBs
- [ ] Apply v2 decision matrix (above) + update results.md with verdict

### Documentation (after LB)

- [ ] Update `MODEL-RUNS.md` with v2 LB scores (replace placeholder section)
- [ ] Update `CONTEXT.md` "What happened recently" section
- [ ] Decision: continue with this direction, or pivot to pseudo-distillation

## What this RULES OUT vs what it LEAVES OPEN (so far)

**Rules out** (or strongly demotes):
- *Nothing yet.* Training succeeded and reproducibility is confirmed; the
  full verdict needs LB.

**Leaves open** (for the LB result to resolve):
- Whether the cmap improvement (+0.011) is a real LB-relevant signal or
  another offline-LB anti-correlation case.
- Whether the higher train loss with similar val/auroc indicates better
  generalization (good for LB) or just under-fitting in a particular
  direction (neutral for LB).
- Whether the Phase A finding generalizes to a different backbone (Phase B).

## Caveats

- **Single fold** for each variant. Adds variance. The +0.011 cmap delta
  is on a single random split (seed=42) — could be noise.
- **Same seed** (42) for both runs. The data ordering, mixup pair selection,
  and crop offsets are identical across the two variants. This is good for
  isolating the mixup effect but means we don't have a seed-noise estimate
  on top.
- **`val_sc` metrics are broken** (exp 026 caveat); listed for completeness
  but don't read into the absolute values.

## Decision matrix (from proposal.md, applied to current state)

| Step | Threshold | v1 status | v2 status |
|---|---|---|---|
| `v5j_control` reproducibility (offline, dualval regime) | val/auroc within ±0.005 of v5j-control 0.851 | ✓ PASS (0.849) | n/a |
| `v5j_control_v2` reproducibility (offline, original regime) | val/auroc within ±0.005 of v5j-original 0.874 | n/a | ✓ PASS (0.873) |
| `v5j_control_v2` LB | should land 0.899-0.906 (dual_val cost recovers) | LB 0.890 | PENDING |
| `v5j_sydorskyi_v2` LB | apply control vs sydorskyi matrix | LB 0.892 (+0.002) | PENDING |
| `effnetv2b0_control_v2` LB | backbone main effect on LB | LB 0.893 (+0.003 vs v5j_control v1) | PENDING |
| `effnetv2b0_sydorskyi_v2` LB | combined effect; ≥ 0.908 = first single-model > v5j 0.906 | LB 0.901 (+0.011 vs v5j_control v1) | PENDING |

## Files

- Configs: `configs/{v5j,effnetv2b0}_{control,sydorskyi}.yaml` (v2 = same files,
  `dual_validation` removed, run_name `-v2` suffix)
- Checkpoints v1: `/storage/project/.../experiments/030-sydorskyi-mixup/checkpoints/2026-exp030-{model}-{recipe}/`
- Checkpoints v2: `/storage/project/.../experiments/030-sydorskyi-mixup/checkpoints/2026-exp030-{model}-{recipe}-v2/`
- Metrics JSON (v2 overwrote v1): `artifacts/training/{v5j,effnetv2b0}_{control,sydorskyi}/metrics.json`
- Test results: `outputs/test_results/2026-exp030-*.json` (v1) and `2026-exp030-*-v2.json` (v2)
- W&B runs: project `birdclef-2026-exp030`
  - v1: `el93xunm`, `3bjn6i3y`, `jhfx75pb`, `bampbee6`
  - v2: `586p423f`, `e6hdxkg8`, `fyqjktl5`, `cg9v7ksj`
- SLURM logs:
  - v1: `slurm_v5j_control_7452610.log`, `slurm_v5j_sydorskyi_7452611.log`,
    `slurm_effnetv2b0_control_7454128.log`, `slurm_effnetv2b0_sydorskyi_7454131.log`
  - v2: `slurm_v5j_control_7455688.log`, `slurm_v5j_sydorskyi_7455689.log`,
    `slurm_effnetv2b0_control_7455690.log`, `slurm_effnetv2b0_sydorskyi_7455691.log`
