# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-12

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/notes/sed-architecture-overview.md` | **Complete pipeline reference** — 13 sections covering data, audio processing, model architecture, training, loss, inference, window experiments, identified gaps | Read fully (primary doc) |
| `user/murilo/notes/2026-04-11-deep-research-claude.md` | Claude deep research: 0.90→0.93 recipe from past winners | Read fully |
| `user/murilo/notes/2026-04-11-deep-research-gemini.md` | Gemini deep research: four systemic pipeline failures | Read fully |
| `user/murilo/experiments/027-model-training-fixes/tasks.md` | **Exp 027 task tracking (ACTIVE)** | Read fully |
| `user/murilo/experiments/027-model-training-fixes/results.md` | Exp 027 results (pending v10a training) | Read fully |
| `user/murilo/experiments/026-domain-shift-training/results.md` | Exp 026 results — dual val + AttHead ablation (all regressed) | Skim |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006–026) | Read fully |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.903** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/027-model-training-fixes` — PR #30

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold** |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.903** | **Best single SED — our ceiling** |
| v5j-dualval (exp 026) | 0.897 | Soundscape checkpoint selection (worse) |
| v5j-pl-v1 (exp 025) | 0.897 | Pseudo-labels (regressed) |
| v9c 20s+AttHead (exp 026) | 0.878 | Longer window + AttHead (regressed) |
| **v10a addmix+CE (exp 027)** | **pending** | **Training — additive MixUp + CE loss fix** |

## What happened in this session (2026-04-12)

### Exp 027: Model Training Fixes — TRAINING COMPLETE, AWAITING LB

Implemented two fixes based on deep research (2026-04-11):
1. **Additive waveform MixUp**: `mixup_mode="additive"` — `x1 + x2` with `max(y1, y2)`
2. **CE loss on primary label**: `loss_type="ce"` on `primary_idx`, sigmoid at inference

**Training results** (job 6439641, RTX 6000, 38m51s):

| Metric | v5j (exp 019) | v5j-dualval (exp 026) | v10a (exp 027) |
|--------|--------------|----------------------|----------------|
| val/auroc | 0.874 | 0.849 | 0.835 |
| val/cmap | — | 0.596 | 0.571 |
| val_sc/auroc | — | 0.956 | 0.863 |
| LB | **0.903** | 0.897 | **pending** |

Focal metrics lower than v5j — expected with CE (different logit calibration).
Only the LB score can tell us if the fixes actually help.

**Checkpoint decision**: Using `last.ckpt` (epoch 19) for ONNX export.
Per-epoch TensorBoard analysis shows val/auroc peaks at epoch 18 (0.839),
epoch 19 within noise (0.835). Matches v5j protocol. Config updated to
monitor val/auroc instead of val_sc/auroc for future runs.

### Git operations

- Squash-merged PR #28 (exp 026) into main
- Created branch `mg/027-model-training-fixes`, opened PR #30
- Recorded v10a metrics, fixed checkpoint monitor, all pushed

## What needs to happen next

### Priority 1: ONNX export + LB submission (next session)

v10a training complete. Offline metrics recorded. Checkpoint decision made:
use `last.ckpt` (epoch 19) — val/auroc peaks at epoch 18 (0.839), epoch 19 is
within noise (0.835). Matches v5j protocol. Config updated to monitor val/auroc
instead of val_sc/auroc for future runs.

1. ONNX export `last.ckpt` via exp 010 pipeline
2. Kaggle submission → LB score vs v5j baseline (0.903)

### Priority 2: Ablation (if v10a result is informative)

If v10a improves or regresses, create ablation configs to isolate contributions:
- v10b: Additive MixUp only (keep BCE loss)
- v10c: CE loss only (keep interpolative spectrogram MixUp)

### Priority 3: Fix inference pipeline (Error 3)

The third fundamental error (train/inference window mismatch) is not addressed
in exp 027. If v10a improves the base model, implement sliding window inference
or Henkel reshape trick to properly evaluate longer training windows.

### Priority 4: Pseudo-labeling with corrected base model

If v10a produces a stronger base model (0.910+), return to exp 025 pseudo-
labeling with the corrected model as teacher.

## Key patterns (updated 2026-04-12)

- **20+ training recipe changes, 0 LB successes** — but deep research revealed
  our implementations were wrong (MixUp, CE loss, inference)
- **Offline metrics are anti-correlated with LB** — DO NOT trust offline metrics
- **The v5j recipe is an island** — every perturbation regresses, but this may
  be because we never fixed the three fundamental errors
- **Exp 027 is testing corrected implementations** — first time testing additive
  MixUp + CE on primary label. Both have strong competition evidence.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/027-model-training-fixes/` | **Active experiment** |
| `user/murilo/experiments/027-model-training-fixes/tasks.md` | Exp 027 task tracking |
| `user/murilo/experiments/027-model-training-fixes/configs/v10a_addmix_ce.yaml` | v10a config |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | Training module (mixup_mode added) |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference (13 sections) |
| `user/murilo/notes/2026-04-11-deep-research-claude.md` | Claude deep research findings |
| `user/murilo/notes/2026-04-11-deep-research-gemini.md` | Gemini deep research findings |
