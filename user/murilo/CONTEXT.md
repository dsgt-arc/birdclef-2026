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
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses -0.031 |
| v5j-10s (10s window, exp 027) | 0.906 | Flat — TTA already covers overlap |

## What happened in this session (2026-04-12)

### Exp 027: Model Training Fixes — v10a FAILED, v10b NEXT

Tested two fixes from deep research (2026-04-11):
1. **Additive waveform MixUp**: `mixup_mode="additive"` — `x1 + x2` with `max(y1, y2)`
2. **CE loss on primary label**: `loss_type="ce"` on `primary_idx`, sigmoid at inference

**v10a LB = 0.494** — catastrophic failure. CE loss produces logits with ~1 unit
spread centered at zero. Sigmoid maps all 234 species to ~0.50 → random on
soundscapes. Root cause: CE optimizes relative ranking (softmax), not absolute
magnitudes. Weight decay keeps logits small. BCE pushes absent species to -9
(robust to domain shift); CE leaves them at -0.2 (destroyed by domain shift).

**Why competitors succeed with CE**: ensembles (min-reduction), pseudo-labels
(bridge domain shift), or calibration. No evidence of CE+sigmoid working as
standalone single-model change with high domain shift.

**CE is now confirmed dead**: 7 tests, 0 successes. See results.md for full
root cause analysis.

**Next**: v10b — additive MixUp + BCE (isolate whether corrected MixUp helps
when paired with the working loss function).

### Git operations

- Squash-merged PR #28 (exp 026) into main
- Created branch `mg/027-model-training-fixes`, opened PR #30
- Recorded v10a metrics, fixed checkpoint monitor, all pushed

## What needs to happen next

### Priority 1: v10b — additive MixUp + BCE (isolate MixUp fix)

v10a combined two changes; CE was catastrophic (0.494) and masks MixUp.
v10b tests additive waveform MixUp with BCE (v5j loss) to isolate whether
the corrected MixUp formulation improves the 0.903 baseline.

Config: identical to v5j except `raw_waveform_mixup=true`, `mixup_mode="additive"`.
Evidence: BirdCLEF 2025 2nd place lost -0.036 without correct MixUp.

1. Create v10b config
2. Train (20 epochs, same as v5j)
3. ONNX export + LB submission

### Priority 2: Overlapping window inference (inference-only)

Both Claude and Gemini identify this as a consistent LB booster (+0.01-0.015).
No retraining needed — modify Kaggle submission kernel only.

Feed 10s context (5s target + 2.5s neighbors from each side), average
overlapping predictions. Can be tested with existing v5j weights.

### Priority 3: Beginning/end crop bias (training change)

Random crops from first or last 7s of focal recordings instead of fully random.
BirdCLEF 2025 2nd place / 2024 3rd place technique. Addresses weak-label noise.

### Priority 4: Pseudo-labeling with best base model

Once single-model improvements are exhausted, return to iterative pseudo-labeling
(exp 025) with the strongest teacher. Pseudo-labels bridge the focal→soundscape
domain shift that limits all training-time changes.

## Key patterns (updated 2026-04-12)

- **20+ training recipe changes, 0 LB successes** — deep research identified
  three errors; CE is now confirmed dead (0.494), MixUp ablation pending
- **CE loss is fundamentally incompatible** — 7 tests, 0 successes. Root cause:
  CE logits lack magnitude spread for sigmoid under domain shift. Competitors
  succeed with ensembles/pseudo-labels, not standalone single-model CE.
- **Offline metrics are anti-correlated with LB** — DO NOT trust offline metrics
- **The v5j recipe is an island** — but additive MixUp (v10b) is untested with BCE
- **Inference-side changes are the highest-value bet** — overlapping windows
  (+0.01-0.015) require no retraining and compound with any training improvement

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
