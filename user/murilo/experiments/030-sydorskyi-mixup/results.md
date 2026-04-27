---
experiment: "030-sydorskyi-mixup"
status: in-progress
date: 2026-04-26
author: Murilo Gustineli
---

# Experiment 030: Sydorskyi-faithful MixUp on v5j — Results

## Status

**Phase A**: training COMPLETE, LB submission PENDING. Both v5j_control and
v5j_sydorskyi finished cleanly. ~37-42 min walltime each on rtx6000.

**Phase B** (EffNetV2-B0 2x2 ablation): not started.

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

## Open work

### Phase A continuation

- [ ] ONNX export for both v5j_control and v5j_sydorskyi
- [ ] Kaggle LB submission for both
- [ ] Apply [proposal.md](proposal.md) decision matrix once LB is in
- [ ] Update results.md with verdict

### Phase B (after Phase A LB results)

- [ ] effnetv2b0_control: SLURM run + LB
- [ ] effnetv2b0_sydorskyi: SLURM run + LB
- [ ] 2x2 ablation analysis: backbone main effect, mixup main effect,
  interaction

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
