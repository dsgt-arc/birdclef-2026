---
experiment: "029-2025-pretrain"
status: in_progress
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

## Stage 2: Fine-tune on 2026 — PENDING

**Config**: `configs/v5j_finetune_2026.yaml` (placeholder
`init.from_checkpoint` updated to point at the stage-1 best checkpoint
above before submission)

Pending steps:
1. Update stage-2 config with stage-1 best checkpoint path (above)
2. Fast-dev-run on stage-2 config to verify transfer logs report
   41 overlap species correctly mapped
3. Full stage-2 training
4. ONNX export
5. LB submission
6. Triage on stage-2 model — does the per-frame structure (entropy,
   logit_std) on `mixed_uncertainty` change vs v5j baseline?

## Comparison to baselines (LB)

To be filled in after stage 2 LB submission.

| Slug | Init | LB | Δ vs v5j |
|------|------|---:|---------:|
| v5j (exp 019) baseline | ImageNet | 0.906 | — |
| v5j-pp2 (exp 021) | ImageNet + priors | 0.909 | +0.003 |
| **exp 029 stage-2 single fold** | **2025-pretrained** | **TBD** | **TBD** |

## Decision-gate state

We're at the boundary between Stage 1 and Stage 2. Stage 1 produced a
clean checkpoint (val/auroc 0.972 at epoch 15) that's ready for transfer.
The decision-gate matrix in [tasks.md](tasks.md) applies to stage-2 LB +
triage, not stage 1.
