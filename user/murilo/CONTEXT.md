# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-12 (end of session)

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/notes/sed-architecture-overview.md` | **Complete pipeline reference** — 13 sections covering data, audio processing, model architecture, training, loss, inference, window experiments, identified gaps | Read fully (primary doc) |
| `user/murilo/notes/2026-04-12-mario-lasseck-birdclef2024.md` | **Mario's BirdCLEF 2024 paper** — iterative pseudo-labeling (+8%), waveform mixing, model soup | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/tasks.md` | **Exp 025 task tracking — Phase C is next** | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/waveform-mixing-blueprint.md` | **Mario-style PL waveform mixing recipe — implementation spec** | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006–027) | Read fully |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |
| `user/murilo/notes/2026-04-12-salman-ahmed-birdclef2025.md` | Salman's BirdCLEF 2025 code analysis (reference) | Skim |
| `user/murilo/experiments/027-model-training-fixes/results.md` | Exp 027 results — all failed (closed) | Skim |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.906** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/025-waveform-mixing-pl` — PR #31

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold** |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | **Best single SED — our ceiling** |
| v5j-dualval (exp 026) | 0.897 | Soundscape checkpoint selection (worse) |
| v5j-pl-v1 (exp 025 Phase B) | 0.897 | Pseudo-labels as separate rows (regressed) |
| v11b simple mel norm (exp 027) | 0.895 | FAILED — mel norm regresses -0.011 |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses -0.031 |
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |

## What happened in this session (2026-04-12)

### Exp 027: Model training fixes — ALL FAILED, CLOSED

Tested 5 changes across 4 phases. Every single one regressed or was flat on LB.
Squash-merged PR #30 into main and closed the experiment.

| Change | LB | Delta vs v5j |
|--------|-----|-------------|
| v10a: additive MixUp + CE loss | 0.494 | -0.412 |
| v10b: additive MixUp + BCE loss | 0.875 | -0.031 |
| v11b: simple mel norm (X+80)/80 | 0.895 | -0.011 |
| v11a: max trick + RMS sampling | — | -0.045 offline |
| I-1: 10s overlapping window inference | 0.906 | +0.000 |

**Running total: 19+ training recipe changes, 0 LB improvements.**

### Set up exp 025 Phase C: waveform-mixing pseudo-labeling

Created branch `mg/025-waveform-mixing-pl` and PR #31.
No implementation yet — next session starts here.

## What needs to happen next

### Priority 1: Implement exp 025 Phase C (waveform-mixing PL)

This is the ONLY remaining direction with evidence of large gains. Mario Lasseck
achieved **+8% from iterative pseudo-labeling** (BirdCLEF 2024 2nd place). Every
top BirdCLEF solution uses pseudo-labeling. Our Phase B failed because we added
PLs as separate dataset rows — Phase C uses Mario's approach of mixing PL audio
INTO focal samples at the waveform level.

**Blueprint**: `025-iter-pseudo-labeling/waveform-mixing-blueprint.md`
**Reference**: `notes/2026-04-12-mario-lasseck-birdclef2024.md`

We already have:
- 347K pseudo-labels from Phase A (196/234 species, 0.929 ensemble teacher)
- Pseudo-labels stored at `shared/pseudo-labels/birdclef-2026/v1/pseudo_labels.parquet`
- Soundscape audio at `shared/data/birdclef-2026/train_soundscapes/`

#### Step 1: Implement `PseudoLabelMixingDataset` in `src/birdclef/data/sed.py`

Wrapper around the existing GT training dataset. On each `__getitem__`:
1. Get normal focal sample (waveform, label, primary_idx) from underlying dataset
2. With probability `pl_mix_chance` (30%), load a random PL segment:
   - Read 5s audio from soundscape file at the segment's start time
   - Apply amplitude factors: `amp = 10**(random.uniform(-0.5, 0.1))`
   - Mix: `mixed = amp_focal * focal_waveform + amp_pl * pl_waveform`
   - Labels: `max(focal_label, pl_soft_labels)`
3. Return mixed (or unmixed) sample as standard 3-tuple

Key: this is a wrapper, NOT a replacement. The underlying dataset and entire v5j
pipeline (mel, MixUp, backbone, head, loss) stay exactly the same.

#### Step 2: Create config and train

- Create `configs/v3_waveform_mix.yaml` — v5j recipe + PL mixing params
- Wire `PseudoLabelMixingDataset` into `SEDDataModule` when config present
- Fast-dev-run sanity check
- Full training (20 epochs) + ONNX export + LB submission
- Compare vs v5j baseline (0.906 LB)

#### Step 3: Iterate (if Step 2 improves)

- Round 2: Use improved student as new teacher → regenerate PLs → retrain
- Round 3: Repeat if still improving (Mario: 3 rounds before diminishing returns)
- 4-fold ensemble of best round → submit with Perch

### Priority 2: Ensemble improvements (lower priority)

- TF Perch instead of ONNX (+0.005 recovery)
- Train v5j with different mel params for ensemble diversity
- Better ensemble weighting (per-species or per-site)

## Key patterns (updated 2026-04-12)

- **19+ training recipe changes, 0 LB successes** — the evidence is overwhelming
  that v5j recipe changes cannot improve LB. Do NOT try more recipe tweaks.
- **The bottleneck is data coverage**: 159 species with zero soundscape data.
  Pseudo-labeling is the only path that directly addresses this.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Max trick
  regresses offline (-0.045). Mel norm regresses on LB (-0.011).
- **Offline metrics remain anti-correlated with LB** — DO NOT trust them.
- **Phase C waveform mixing is fundamentally different from all previous failures**:
  every failed change modified HOW the model learns from the SAME data. Phase C
  changes WHAT data the model sees — 127K soundscape segments it never had.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/025-iter-pseudo-labeling/` | **Active experiment** |
| `user/murilo/experiments/025-iter-pseudo-labeling/tasks.md` | Task tracking — Phase C next |
| `user/murilo/experiments/025-iter-pseudo-labeling/waveform-mixing-blueprint.md` | Implementation spec |
| `user/murilo/notes/2026-04-12-mario-lasseck-birdclef2024.md` | Mario's approach reference |
| `user/murilo/src/birdclef/data/sed.py` | Data module — add `PseudoLabelMixingDataset` here |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | Training module (no changes needed) |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference (13 sections) |
| `user/murilo/FINDINGS.md` | What worked / didn't across all experiments |
| `shared/pseudo-labels/birdclef-2026/v1/pseudo_labels.parquet` | 347K pseudo-labels from Phase A |
