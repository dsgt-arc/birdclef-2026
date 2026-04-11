# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-11

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/notes/sed-architecture-overview.md` | **Complete pipeline reference** — 13 sections covering data, audio processing, model architecture, training, loss, inference, window experiments, identified gaps | Read fully (primary doc) |
| `user/murilo/notes/2026-04-11-deep-research-claude.md` | Claude deep research: 0.90→0.93 recipe from past winners | Read fully |
| `user/murilo/notes/2026-04-11-deep-research-gemini.md` | Gemini deep research: four systemic pipeline failures | Read fully |
| `user/murilo/experiments/026-domain-shift-training/results.md` | Exp 026 results — dual val + AttHead ablation (all regressed) | Read fully |
| `user/murilo/experiments/026-domain-shift-training/tasks.md` | Exp 026 task tracking | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006–024) | Read fully |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim exp 026 section |
| `user/murilo/experiments/025-iter-pseudo-labeling/tasks.md` | Exp 025 pseudo-labeling (paused — round 1 regressed) | Skim |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.903** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/026-domain-shift-training` — PR #28

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold** |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.903** | **Best single SED — our ceiling** |
| v5j-dualval (exp 026) | 0.897 | Soundscape checkpoint selection (worse) |
| v5j-pl-v1 (exp 025) | 0.897 | Pseudo-labels (regressed) |
| v9c 20s+AttHead (exp 026) | 0.878 | Longer window + AttHead (regressed) |
| v9b 15s+AttHead (exp 026) | 0.877 | Longer window + AttHead (regressed) |
| v9a 10s+AttHead (exp 026) | 0.868 | Longer window + AttHead (regressed) |

## What happened in this session (2026-04-11)

### 1. Exp 025: Pseudo-labeling round 1 — REGRESSED (0.897)

Trained v5j with 347K pseudo-labels (SED + Perch z-norm), 50/50 batch mixing,
w=0.5 loss weight. LB: 0.897 (−0.006 from baseline 0.903).
Created diagnosis pipeline (`diagnose_regression.py`) and v2-sed-only configs.
**Paused** — pivoted to improving base model first.

### 2. Exp 026: Domain-shift training — ALL REGRESSED

**Phase 2: Dual validation** — infrastructure works (separate focal/soundscape
metrics), but soundscape checkpoint selection picked epoch 17 instead of 19,
scoring 0.897 LB (−0.006). Soundscape AUC not useful for checkpoint selection
with only 86 val segments.

**Phase 3: AttHead + window ablation** — ALL regressed on LB:
- v9a (10s + AttHead): 0.868 (−0.035)
- v9b (15s + AttHead): 0.877 (−0.026)
- v9c (20s + AttHead): 0.878 (−0.025)

Despite best-ever offline metrics (v9c: 0.861 auroc, 0.681 cmap, 0.862 top1).
**Root cause identified**: train/inference window mismatch — train on 20s but
infer on 5s segments. AttHead's learned attention calibrated for 64 temporal
positions doesn't transfer to 16 positions at inference.

### 3. Deep Research — CRITICAL FINDINGS

Ran deep research with Claude Opus 4.6 and Gemini Pro. Both converge on the
same diagnosis: **three fundamental implementation errors** in our pipeline.

#### Error 1: Waveform MixUp was wrong

We used `λ*x1 + (1-λ)*x2` (standard image MixUp). BirdCLEF winners use
**additive mixing** (`x1 + x2`) with **max labels** (`max(y1, y2)`).
Completely different operation. Spectrogram MixUp also violates physics of
additive sound waves. Impact: −0.036 private LB (BirdCLEF 2025 2nd place ablation).

#### Error 2: CE loss implementation was wrong

We applied CE as multi-class softmax across all 234 species with multi-hot targets.
The correct approach: standard **multi-class CE on primary label only** (single
ground-truth class per sample, softmax over classes). Then switch to **sigmoid at
inference** for multi-label output. BirdCLEF 2024 1st place: +0.044 from this
single change.

#### Error 3: Inference pipeline never matched training window

We always process 5s segments independently on Kaggle. Winners use:
- **Sliding window inference**: 20s windows with 1–2.5s stride, average overlapping
  framewise predictions for each 5s evaluation segment
- **Henkel reshape trick**: train on 30s but reshape to 6×5s before backbone —
  backbone always sees 5s, no mismatch at inference

#### Additional findings

- **Crop selection**: first/last 7 seconds of focal recordings (not random, not RMS)
  because recordings start/stop when the target animal is vocalizing (BirdCLEF 2025
  2nd place). Gemini advocates RMS-based VAD — this is disputed.
- **Iterative pseudo-labeling works**: BirdCLEF 2025 1st place went 0.872 → 0.930
  with 4 rounds. Our attempt failed but with wrong base model + wrong parameters.
- **PANNs AttBlock is the standard head**: competition-standard attention with
  `tanh + softmax`, should be length-agnostic (no positional encoding)

## What needs to happen next

### Priority 1: Fix the three implementation errors

These are code changes, not new experiments. Each should be testable independently:

1. **Fix waveform MixUp**: additive mixing (`x1 + x2`) with max labels
2. **Fix CE loss**: multi-class CE on primary label, sigmoid at inference
3. **Fix inference pipeline**: sliding window with overlap, or Henkel reshape

### Priority 2: Re-test with corrected pipeline

Once the implementation errors are fixed:
- Train v5j recipe with corrected MixUp + CE loss
- Submit with corrected inference pipeline
- This should close a significant portion of the 0.903 → 0.922 gap

### Priority 3: Then pseudo-labeling

With a stronger base model (hopefully 0.910+), return to exp 025 pseudo-labeling
with a better teacher model. The iterative approach (student becomes teacher)
is validated by 2025 1st place.

## Key patterns (updated 2026-04-11)

- **20+ training recipe changes, 0 LB successes** — but deep research revealed
  our implementations were wrong (MixUp, CE loss, inference)
- **Offline metrics are anti-correlated with LB** — best offline models (v9c)
  scored worst on LB (0.878). DO NOT trust offline metrics.
- **The v5j recipe is an island** — every perturbation regresses, but this may
  be because we never fixed the three fundamental errors
- **The inference pipeline is likely our biggest blind spot** — we never changed
  inference to match training window. This alone could explain -0.025 regression.
- **Competitors don't use novel architectures** — EfficientNet + PANNs AttBlock
  is the standard. The gap is in data handling, loss function, and inference.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference (13 sections) |
| `user/murilo/notes/2026-04-11-deep-research-claude.md` | Claude deep research findings |
| `user/murilo/notes/2026-04-11-deep-research-gemini.md` | Gemini deep research findings |
| `user/murilo/notes/deep-research-prompt.md` | The prompt used for deep research |
| `user/murilo/experiments/026-domain-shift-training/` | Current experiment (dual val + AttHead) |
| `user/murilo/experiments/025-iter-pseudo-labeling/` | Pseudo-labeling experiment (paused) |
| `user/murilo/src/birdclef/data/sed.py` | Data module (dual val, pseudo-labels) |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | Training module (loss, metrics) |
| `user/murilo/src/birdclef/heads/sed_head.py` | LSEHead, AttHead implementations |
