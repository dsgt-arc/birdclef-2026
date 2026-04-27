---
experiment: "030-sydorskyi-mixup"
status: in-progress
date: 2026-04-26
author: Murilo Gustineli
---

# Experiment 030: Sydorskyi-faithful MixUp on v5j — Results

## Status

**v1 (with `dual_validation: true`)**: training + LB COMPLETE. Re-training in
progress with `dual_validation` removed (v2) to recover the original v5j
0.906 LB regime.

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

## Combined offline picture (all 4 cells)

The Sydorskyi mixup recipe and EfficientNetV2-B0 backbone both improve our
offline cmap/f1_macro metrics, and they stack roughly additively. The best
cell (`effnetv2b0_sydorskyi`) hits val/cmap 0.6303 — a +0.034 improvement
over v5j (0.5959) on cmap.

But this is offline. The historical offline-LB anti-correlation is the
strong prior (val/auroc 0.874 → 0.906 LB for v5j, vs val/auroc 0.874 →
0.899 LB for v6l manifold mixup). cmap improvements have not consistently
translated to LB gains.

## Open work

### Phase A continuation

- [ ] ONNX export for both v5j_control and v5j_sydorskyi
- [ ] Kaggle LB submission for both
- [ ] Apply [proposal.md](proposal.md) decision matrix once LB is in
- [ ] Update results.md with verdict

### Phase B (training DONE; LB pending)

- [x] effnetv2b0_control: SLURM run COMPLETE (val/auroc 0.8510)
- [x] effnetv2b0_sydorskyi: SLURM run COMPLETE (val/auroc 0.8518 — best cell)
- [x] 2x2 offline ablation analysis: backbone +0.020 cmap, mixup +0.014 cmap,
  mostly additive interaction
- [ ] effnetv2b0_control: ONNX export + LB
- [ ] effnetv2b0_sydorskyi: ONNX export + LB
- [ ] Apply 2x2 LB analysis (main effects + interaction on LB)

### Documentation (after LB)

- [ ] Update `MODEL-RUNS.md` with LB scores
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

| Step | Threshold | Status |
|---|---|---|
| `v5j_control` reproducibility | val/auroc within ±0.005 of v5j-control 0.851 | ✓ PASS (0.849) |
| `v5j_control` LB | should land 0.901-0.911 | PENDING |
| `v5j_sydorskyi` LB | apply control vs sydorskyi matrix | PENDING |

## Files

- Configs: `configs/v5j_control.yaml`, `configs/v5j_sydorskyi.yaml`
- Checkpoints: `/storage/project/.../experiments/030-sydorskyi-mixup/checkpoints/`
- Metrics JSON: `artifacts/training/v5j_*/metrics.json`
- Test results: `outputs/test_results/2026-exp030-*.json`
- W&B runs: project `birdclef-2026-exp030`
- SLURM logs: `logs/slurm_v5j_control_7452610.log`, `logs/slurm_v5j_sydorskyi_7452611.log`
