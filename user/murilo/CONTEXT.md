# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-12 (end of session)

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/notes/sed-architecture-overview.md` | **Complete pipeline reference** — 13 sections covering data, audio processing, model architecture, training, loss, inference, window experiments, identified gaps | Read fully (primary doc) |
| `user/murilo/notes/2026-04-12-salman-ahmed-birdclef2025.md` | **Salman's BirdCLEF 2025 code analysis** — max trick, RMS sampling, mel norm, full pipeline comparison vs v5j | Read fully |
| `user/murilo/notes/2026-04-12-mario-lasseck-birdclef2024.md` | **Mario's BirdCLEF 2024 paper** — iterative pseudo-labeling (+8%), waveform mixing, model soup | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/tasks.md` | **Exp 025 task tracking — Phase C next (waveform mixing)** | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/waveform-mixing-blueprint.md` | **Mario-style PL waveform mixing recipe** | Read fully |
| `user/murilo/experiments/027-model-training-fixes/tasks.md` | Exp 027 task tracking — Phases 1-4 (winding down) | Read fully |
| `user/murilo/experiments/027-model-training-fixes/results.md` | Exp 027 results — all variants | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006–027) | Read fully |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.906** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/025-waveform-mixing-pl` — PR pending

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold** |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | **Best single SED — our ceiling** |
| v5j-dualval (exp 026) | 0.897 | Soundscape checkpoint selection (worse) |
| v5j-pl-v1 (exp 025) | 0.897 | Pseudo-labels (regressed) |
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses -0.031 |
| v11b simple mel norm (exp 027) | 0.895 | FAILED — mel norm regresses -0.011 |
| v5j-10s (10s window, exp 027) | 0.906 | Flat — TTA already covers overlap |

## What happened in this session (2026-04-12)

### Exp 027 Phases 1-3: Deep research recommendations — ALL FAILED

Tested three changes from Claude+Gemini deep research (2026-04-11):

| Change | LB | Delta | Outcome |
|--------|-----|-------|---------|
| v10a: additive MixUp + CE loss | 0.494 | -0.412 | CE logits random on soundscapes |
| v10b: additive MixUp + BCE loss | 0.875 | -0.031 | Additive MixUp worse than spectrogram |
| I-1: 10s overlapping window inference | 0.906 | +0.000 | TTA already covers overlap |

CE confirmed dead (7 tests). Additive MixUp confirmed worse (3 waveform MixUp tests, all regress).

### Exp 027 Phase 4: Salman-derived structural changes

Analyzed Salman Ahmed's BirdCLEF 2025 training notebook and Mario Lasseck's
BirdCLEF 2024 paper. Identified specific implementation differences from v5j.
Created comprehensive reference notes for both.

| Change | val/auroc | Delta vs control | Notes |
|--------|-----------|-----------------|-------|
| v5j-control (baseline) | 0.849 | — | Matches v5j-dualval, pipeline verified |
| v11a: max trick + RMS | 0.804 | **-0.045** | Focal regression, val_sc flat |
| v11b: simple mel norm | 0.847 | -0.002 | **LB 0.895 (-0.011)** |

v11a (max trick) regressed substantially on focal metrics despite being Salman's
most distinctive technique. Likely cause: train/val mismatch (RMS-selected training
vs random val crops) and reduced batch diversity (32 vs 64 samples).

### Key realization: the bottleneck is data coverage, not model quality

After 27 experiments and 18+ training changes with 0 LB improvements:

- v5j achieves **0.9958 AUC** on 75 species seen in soundscapes (near-perfect)
- **159 species** have zero soundscape training data (mean max-prob 0.036)
- The entire LB gap comes from species the model never saw in soundscapes
- No training recipe change can fix this — only DATA that bridges the domain gap

### Git operations

- Squash-merged PR #28 (exp 026) into main
- All Phase 4 code (max trick, RMS sampling, simple_range mel norm) implemented
- Salman + Mario reference notes created
- v5j-control, v11a, v11b metrics recorded

## What needs to happen next

### Priority 1: Mario-style pseudo-labeling (exp 025 Phase C)

The ONLY direction with evidence of large gains (+8% from Mario, used by every
top BirdCLEF solution). Extending exp 025 (not new experiment) since PL infra
already exists. Blueprint: `025-iter-pseudo-labeling/waveform-mixing-blueprint.md`

**Mario's approach** (BirdCLEF 2024 2nd place):
1. Mix pseudo-labeled soundscape AUDIO with focal audio at waveform level
2. Amplitude factors: `ampFactor = 10**(random.uniform(-0.5, 0.1))`
3. Labels: `max(focal_label, pseudo_label_probs)`
4. 25-45% of training samples get mixed with PL segments
5. Iterate 2-3 times (retrain → regenerate PLs → retrain)

We already have: 347K pseudo-labels from Phase A, 0.929 ensemble as teacher.
What failed in Phase B: adding PL as separate dataset rows (wrong approach).

Implementation needed:
1. `PseudoLabelMixingDataset` wrapper class in `src/birdclef/data/sed.py`
2. `v3_waveform_mix.yaml` config
3. Train + ONNX export + LB submission

### Priority 2: v11b LB score — DONE (0.895, regressed -0.011)

v11b (simple mel norm) scored **0.895 LB** (-0.011 vs v5j 0.906). Another training
recipe change that regresses. Mel normalization change is now a confirmed dead end.

### Priority 3: Ensemble improvements

- TF Perch instead of ONNX (+0.005 recovery)
- Train v5j with different mel params for ensemble diversity
- Better ensemble weighting (per-species or per-site)

## Key patterns (updated 2026-04-12, end of session)

- **18+ training recipe changes, 0 LB successes** — including max trick, RMS
  sampling, additive MixUp, CE loss, overlapping windows, mel normalization.
  The evidence is now overwhelming: v5j recipe changes cannot improve LB.
- **The bottleneck is data coverage**: 159 species with zero soundscape data.
  Pseudo-labeling is the only path that directly addresses this.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Max trick
  regresses offline (-0.045). Only spectrogram MixUp works.
- **Offline metrics remain anti-correlated with LB** — DO NOT trust them.
- **Mario's waveform-mixing pseudo-label approach is the clear next step** —
  fundamentally different from our failed exp 025 (mixing audio, not adding rows).

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/027-model-training-fixes/` | **Active experiment** |
| `user/murilo/experiments/027-model-training-fixes/tasks.md` | Exp 027 task tracking (Phases 1-4) |
| `user/murilo/notes/2026-04-12-salman-ahmed-birdclef2025.md` | Salman's pipeline analysis |
| `user/murilo/notes/2026-04-12-mario-lasseck-birdclef2024.md` | Mario's pseudo-labeling paper |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | Training module (max_trick, mixup_mode) |
| `user/murilo/src/birdclef/data/sed.py` | Data module (rms_sampling) |
| `user/murilo/src/birdclef/preprocessing/nnaudio_mel.py` | Mel transform (simple_range norm) |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference (13 sections) |
| `user/murilo/FINDINGS.md` | What worked / didn't across all experiments |
