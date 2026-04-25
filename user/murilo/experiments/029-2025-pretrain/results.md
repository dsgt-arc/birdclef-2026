---
experiment: "029-2025-pretrain"
status: rejected
date: 2026-04-25
author: Murilo Gustineli
---

# Experiment 029: 2025 Pretrain → 2026 Fine-tune — Results

## Context

Sequential transfer-learning experiment motivated by the 2026-04-25
[per-frame analysis](../../analysis/notes/2026-04-25-per-frame-attention-analysis.md):
v5j is structurally honest on hard crops, so further training-recipe
tweaks won't move LB. Adding more *information* (Ali Ozan's 2025 pretrain
path → 0.941 LB) is the credible remaining lever.

**Hypothesis**: pretraining v5j on BirdCLEF 2025 focal data, then
fine-tuning on 2026 with the 41 overlap species' classifier rows
transferred, will improve LB over v5j (0.906) — single-variable change
from v5j (2025-pretrained init vs ImageNet init).

See [proposal.md](proposal.md) for full hypothesis, method, scoping
decisions, and limitations.

## Stage 1: Pretrain on 2025 — COMPLETE

**Config**: `configs/v5j_pretrain_2025.yaml`
**Changes from v5j**: `data_dir` → 2025 path (28,564 rows, 206 species, no
labeled soundscapes), `dual_validation: false`, `soundscape_val_files: 0`.
Otherwise identical to v5j.

**SLURM**: Job 7432887, RTX 6000, **30m16s walltime** (way under the 1.5–2h estimate)
**Checkpoint**: epoch 15 (best by val/auroc)
**Path**: `/storage/.../experiments/029-2025-pretrain/checkpoints/2026-exp029-v5j-pretrain-2025/best-epochepoch=15.ckpt`

### Final metrics (epoch 15)

| Metric | Value |
|--------|------:|
| **val/auroc** | **0.972** |
| val/cmap | 0.649 |
| val/f1_macro | 0.502 |
| val/f1_micro | 0.709 |
| val/top1_acc | 0.761 |
| val/top5_acc | 0.869 |
| val/loss | 0.011 |

Test set not run for stage 1 (skipped — pretrain stage is for representation
learning; final evaluation is at stage 2 LB submission).

### Training trajectory

Best by val/auroc each epoch (epochs where val/auroc was not in top 1
omitted from the sequence — model didn't beat its prior best):

| Epoch | val/auroc | Δ vs prior best | Note |
|------:|----------:|----------------:|------|
| 0 | 0.543 | — | After warmup epoch |
| 1 | 0.587 | +0.044 | |
| 2 | 0.794 | +0.207 | Rapid representation learning kicks in |
| 3 | 0.913 | +0.119 | |
| 4 | 0.934 | +0.021 | |
| 5 | 0.945 | +0.011 | MixUp warmup ends |
| 6 | 0.953 | +0.007 | |
| 7 | 0.959 | +0.006 | |
| 8 | 0.962 | +0.003 | |
| 10 | 0.963 | +0.001 | (epoch 9 didn't beat) |
| 11 | 0.968 | +0.005 | |
| 13 | 0.970 | +0.002 | (epoch 12 didn't beat) |
| **15** | **0.972** | **+0.002** | **Best — checkpoint saved.** (epoch 14 didn't beat) |
| 16–19 | (no improvement) | 0 | Plateau through end of OneCycleLR schedule |

### Observations

1. **Convergence shape was clean.** Steady climb from 0.543 → 0.972 over 15
   epochs, then plateau. No overfitting (val/auroc didn't drop), no
   instability.

2. **Comparable to focal-only training scale.** v5j (2026 focal + labeled
   soundscapes) reached val/auroc ~0.874 — but that mixes in
   harder-to-learn soundscape segments. 2025 focal-only is structurally
   easier (recording-level labels with single dominant species), so
   ~0.97 is in the expected range for focal-only val/auroc.

3. **OneCycleLR(20 epochs) was sufficient for convergence on this dataset.**
   The schedule's high-LR phase (epochs 5–10) drove the bulk of learning;
   the cool-down phase (epochs 10–20) refined slowly. Epochs 16–19 added
   no measurable gain.

### On training longer — should we have used > 20 epochs?

Murilo flagged this concern. Honest answer: probably not at the current
schedule shape. The model was clearly plateauing at epoch 15 and remained
flat through epoch 19. With OneCycleLR(20 ep), the LR is well into
cool-down by epoch 15, so additional epochs at this schedule wouldn't
have helped.

**However**, there's a different question worth flagging: would a
*differently-shaped* schedule (e.g., 30 epochs with stretched OneCycleLR,
or cosine annealing instead) have produced richer representations for
transfer? The current data can't answer that. If stage 2 underperforms
v5j, "use a longer pretrain schedule" is a candidate retry.

## Stage 2: Fine-tune on 2026 — TRAINING COMPLETE, LB PENDING

**Config**: `configs/v5j_finetune_2026.yaml` with
`init.from_checkpoint` pointed at stage-1 best checkpoint
(`best-epochepoch=15.ckpt`).

**Classifier transfer at init** (verified in fast-dev-run logs):
- 318 full-copy params (backbone + head except final classifier)
- 2 partial-copy params (`sed_head.cls_fc.3.weight`, `sed_head.cls_fc.3.bias`)
- 41 of 234 target species mapped from source
- 0 skipped — no pretrained params lost

**SLURM**: Job 7433914, RTX 6000, **38m40s walltime**
**Best checkpoint**: epoch 12 (best by val/auroc)
**Path**: `/storage/.../experiments/029-2025-pretrain/checkpoints/2026-exp029-v5j-finetune-2026/best-epochepoch=12.ckpt`

### Final metrics (epoch 12 = best)

| Metric | Value |
|--------|------:|
| **val/auroc** | **0.850** |
| val/cmap | 0.590 |
| val/f1_macro | 0.505 |
| val/top1_acc | 0.757 |
| val/top5_acc | 0.872 |
| val_sc/auroc | 0.950 (broken metric — see exp 026 caveat) |
| val/loss | 0.011 |

### Training trajectory

| Epoch | val/auroc | Δ vs prior best | Note |
|------:|----------:|----------------:|------|
| 0 | 0.566 | — | Warm start: head already calibrated for 41 species |
| 1 | 0.635 | +0.069 | |
| 2 | 0.737 | +0.103 | |
| 3 | 0.800 | +0.063 | |
| 4 | 0.820 | +0.020 | |
| 5 | 0.829 | +0.009 | MixUp warmup ends |
| 6 | 0.830 | +0.001 | |
| 7 | 0.833 | +0.003 | |
| 8 | 0.837 | +0.004 | |
| 9 | 0.843 | +0.006 | |
| 11 | 0.849 | +0.005 | (epoch 10 didn't beat) |
| **12** | **0.850** | **+0.001** | **Best — checkpoint saved.** |
| 13–19 | (no improvement) | 0 | Plateau through end of OneCycleLR schedule |

### Comparison vs comparable baseline (apples-to-apples)

The original v5j (exp 019, no dual_validation) reports val/auroc 0.874,
but that metric is computed across focal+SS combined and is not
directly comparable. The valid comparison is **v5j-control (exp 027,
dual_validation enabled)**, which gives a focal-only val/auroc:

| Slug | Init | Best val/auroc | Best epoch | Δ |
|---|---|---:|---:|---:|
| v5j-control (exp 027) | ImageNet | 0.851 | ~14 | — |
| **exp 029 stage 2** | **2025-pretrained** | **0.850** | **12** | **−0.001** |

**Within seed noise.** The 2025 pretrain produced a model essentially
identical on offline focal val/auroc to v5j-control. The 41-species
warm start gave an early head start (epoch 0: 0.566 vs ImageNet
~0.5something) but this washes out by the time training equilibrates
across all 234 classes.

### Observations

1. **Best epoch shifted earlier (12 vs ~14).** The warm-started model
   plateaus a bit sooner. Consistent with the 41 overlap species
   already being calibrated.

2. **Reduced LR (2e-4 vs v5j's 5e-4) didn't hurt the final val/auroc
   noticeably.** Confirms that the lower LR was a defensible warm-start
   choice; whether it's optimal vs full LR is empirically untested in
   this run.

3. **Offline metric being neutral does not predict LB.** Throughout this
   project, offline val/auroc has been anti-correlated or uncorrelated
   with LB. The fact that exp 029 stage-2 matches v5j-control offline
   tells us nothing about whether LB will move.

## Stage 3: ONNX export + LB submission — DONE (REGRESSED)

- ONNX export: completed
- Kaggle dataset slug: `birdclef-2026-hgnet-b0-sed-v5j-finetune-weights`
  (47 chars; original 52-char slug exceeded Kaggle's 50-char limit and
  silently failed to attach to the kernel — fixed mid-flight)
- LB result: **0.885** (Public LB)

### LB comparison

| Slug | Init | LB | Δ vs v5j |
|------|------|---:|---------:|
| v5j (exp 019) baseline | ImageNet | **0.906** | — |
| v5j-pp2 (exp 021) | ImageNet + priors | 0.909 | +0.003 |
| v5j-4fold-pp2 (exp 021, 4-fold ensemble) | ImageNet + priors | 0.917 | +0.011 |
| **exp 029 stage-2 single fold** | **2025-pretrained** | **0.885** | **−0.021** |
| Reference: Ali Ozan single (filtered XC pretrain) | XC-pretrained | 0.941 | (different pretrain corpus, much larger) |

### Per-frame triage on stage-2 checkpoint — pending (independent)

`precompute_losses.py` re-run on stage-2 ckpt + `per_frame_analysis.py`
comparison to v5j baseline still queued. Not blocking the headline
result, but useful for understanding *what* changed structurally.

## Decision-gate state: REJECT

Decision matrix from [tasks.md](tasks.md):

| Outcome | LB | Action |
|---------|---|---|
| Confirm | > 0.906 | — |
| Strong confirm | ≥ 0.910 | — |
| Retry | ≤ 0.906 with per-frame structural shift | — |
| **Reject** | **≤ 0.906 by ≥0.005** | **2025 pretrain at our scale doesn't transfer; consider Salman replication or silent-class as next exp** |

LB delta is **−0.021**, well past the −0.005 threshold for "Reject".
Per-frame triage hasn't run yet, but the LB signal is decisive on its
own.

## Interpretation

Three things are simultaneously true:

1. **The pipeline worked end-to-end.** Stage 1 converged cleanly
   (val/auroc 0.972). Stage 2 trained without instability (val/auroc
   0.850). Classifier transfer logged the expected 41 species mapped.
   ONNX export and Kaggle inference ran cleanly. So this is a clean
   negative result, not a broken experiment.

2. **Offline did not predict LB direction** (yet again). val/auroc
   0.850 was within ±0.001 of v5j-control's 0.851. LB regressed
   −0.021. The offline-LB anti-correlation pattern continues — this
   is at least the 22nd training-side change where offline metrics
   gave no LB signal.

3. **The result is consistent with the 2026-04-25 per-frame finding.**
   That note argued v5j sits at a generalization equilibrium and
   training-side changes that perturb that equilibrium tend to hurt
   LB. Pretraining is a training-side perturbation — different
   initialization, different LR (2e-4 vs 5e-4), different feature
   space at the start. The −0.021 result is what the structural
   argument predicted.

### Possible direct mechanisms

Without per-frame triage we can't fully attribute, but candidates:

- **Reduced LR (2e-4 vs v5j's 5e-4)** undertrained the 193 random-init
  classifier rows. They reached a worse equilibrium for soundscape
  inference. Standard transfer-learning advice (lower LR for warm
  start) doesn't fit cleanly here because most of the head was
  random-init, not fine-tuned.
- **Mixed-init head** (41 warm + 193 random) creates an asymmetric
  optimization landscape. Warm rows pull gradients in directions that
  don't help the random rows learn from 2026 data alone.
- **Negative transfer from 2025 backbone features.** 2025 species
  composition differs from 2026; the backbone may have learned
  emphases that hurt 2026 species the model has to un-learn at lower
  LR than v5j.
- **Recipe perturbation** generally — per-frame finding says v5j sits
  at a generalization equilibrium; *any* training trajectory change
  (different init, different LR, different optimizer dynamics) can
  leave the model at a worse equilibrium for soundscape inference
  even when offline focal val/auroc looks identical.

The cleanest single-variable retry would be `head_lr: 5e-4` with
`lr: 2e-4` (split LR — lower backbone, full head), testing whether
under-training the random-init head is the dominant cause. But
strategically the question is whether to keep investing in
training-side variations at all given the per-frame finding.

## What this rules out and what it leaves open

**Ruled out (or strongly demoted):**
- 2025 pretrain at this scale (28K rows, 41-species overlap, with
  reduced LR for warm start) does not transfer usefully to 2026 LB.
- Standard transfer-learning practice (lower LR for warm start) does
  not work cleanly when most of the classifier is random-init.

**Still open (cheap single-axis retries):**
- `head_lr: 5e-4` / `lr: 2e-4` (split LR) — diagnoses whether
  random-init head was undertrained. ~38 min training run.
- Filtered XC pretrain (matching Ali Ozan more faithfully) — much
  larger and richer pretrain corpus, but significant data engineering
  required.

**Still open (different bets):**
- exp 030 silent-class — engages the structural finding directly.
- Salman recipe replication on EffNetV2-B0 — different architecture
  bet; only the disclosed parts are replicable (the +0.015 attention
  jump remains opaque).

## Open work after LB result

- Run per-frame triage on stage-2 checkpoint (small SLURM job already
  queued for user). Independent of the LB headline; tells us *what*
  changed structurally even though LB regressed.
- Decide next experiment direction — see strategic discussion in the
  session.
