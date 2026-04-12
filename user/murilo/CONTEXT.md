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

### Exp 027: Model Training Fixes — IN PROGRESS

Based on deep research findings (2026-04-11), implemented two fixes to the
training recipe that both Claude and Gemini identified as fundamental errors:

**Fix 1: Additive waveform MixUp**
- Added `mixup_mode="additive"` parameter to `SEDFinetuneModule` (backward
  compatible, default `"interpolate"` preserves all existing experiments)
- Changed from `λ*x1 + (1-λ)*x2` with weighted labels to `x1 + x2` with
  `max(y1, y2)` — physically correct audio superposition
- Applied on raw waveforms before mel conversion (`raw_waveform_mixup=true`)

**Fix 2: CE loss on primary label**
- Config uses existing `loss_type="ce"` which applies `F.cross_entropy` on
  `primary_idx` (single ground-truth class per sample, softmax over 234 species)
- Inference already uses sigmoid via `_blended_probs` — this is the CE-train /
  sigmoid-infer paradigm from BirdCLEF 2024 1st place (+0.044 LB)

**Fast-dev-run verified** (job 6439254): CE loss=5.454 (matches log(234)≈5.46
for random softmax init), both val dataloaders fire, waveform mixup before mel
confirmed. Everything working correctly.

**Full training run submitted** — awaiting completion. Next: check metrics,
ONNX export, Kaggle submission.

### Git operations

- Committed + pushed exp 026 artifacts, squash-merged PR #28 into main
- Created branch `mg/027-model-training-fixes` from main
- Committed exp 027 scaffold + src change, pushed, opened PR #30

## What needs to happen next

### Priority 1: Check v10a training results (next session)

1. Check SLURM logs for v10a training completion
2. Record offline metrics (val/auroc, val_sc/auroc, cmap, top1)
3. ONNX export the best checkpoint
4. Kaggle submission → LB score vs v5j baseline (0.903)

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
