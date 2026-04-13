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
| v3-waveform-mix (exp 025 Phase C) | 0.894 | FAILED — Mario-style PL waveform mixing -0.012 |
| v11b simple mel norm (exp 027) | 0.895 | FAILED — mel norm regresses -0.011 |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses -0.031 |
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |

## What happened in this session (2026-04-13)

### Exp 025 Phase C: Waveform mixing — FAILED (0.894 LB, -0.012)

Implemented Mario Lasseck's waveform-mixing approach: 30% of focal samples get
overlaid with random PL soundscape segment (amplitude-controlled). Trained 40 epochs,
model still improving at epoch 38. val_sc/auroc=0.991 but LB regressed -0.012.

| Approach | LB | Delta vs v5j |
|----------|-----|-------------|
| v5j baseline | 0.906 | — |
| Phase B: row-addition (w=0.5) | 0.897 | -0.009 |
| Phase C: waveform mixing (30%) | 0.894 | -0.012 |

**Both pseudo-label approaches failed.** Waveform mixing was worse than row-addition.

### Re-evaluation: Deep research in progress

Key realization: the 0.906→0.93+ gap vs competitors (Boredom 0.938, Salman 0.922)
using the **same labeled data** is the real problem. We've been trying to add new
data via PLs when we're not extracting full value from labeled data.

**Running Gemini Deep Research + Claude Deep Research** with an unbiased prompt
(no context of our experiments) to get fresh perspective on what separates 0.90-level
models from 0.93+ models. Will synthesize findings into new experiment plan.

**Running total: 21+ changes (19 recipe + 2 PL), 0 LB improvements.**

## What needs to happen next

### Priority 1: Synthesize deep research findings

Gemini and Claude deep research running on BirdCLEF 2026 strategy (unbiased, no
context of our experiments). Cross-reference their recommendations, identify
convergent themes, and build evidence-based plan for next experiments.

### Priority 2: Understand where we're failing

Before trying more techniques, need diagnostic analysis:
- Where does our model fail on soundscapes? (per-species, per-site, SNR)
- Why does every v5j modification regress?
- What are Boredom/Salman doing differently for +0.016-0.032 from same data?

### Priority 3: New experiment based on findings

TBD — waiting for deep research results.

## Key patterns (updated 2026-04-13)

- **21+ changes, 0 LB successes** — includes 19 training recipe changes AND 2
  pseudo-label approaches. The data coverage hypothesis was wrong: adding soundscape
  data via PLs didn't help either.
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.922) beat us by
  0.016-0.032 using the same focal + labeled soundscape data. No KD, no Perch.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Pseudo-labels
  are dead (2 approaches). Mel norm, max trick, multi-context all regressed.
- **Offline metrics remain anti-correlated with LB** — DO NOT trust them.
- **Need fresh perspective** — deep research in progress to break out of local minimum.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/025-iter-pseudo-labeling/` | Exp 025 — Phase C failed |
| `user/murilo/experiments/025-iter-pseudo-labeling/results.md` | Full Phase B+C results |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference |
| `user/murilo/FINDINGS.md` | What worked / didn't across all experiments |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints |
