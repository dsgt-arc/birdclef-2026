# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-13 (end of session)

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/experiments/028-domain-shift-augmentation/proposal.md` | **Exp 028 proposal — next experiment** | Read fully (start here) |
| `user/murilo/experiments/028-domain-shift-augmentation/tasks.md` | **Exp 028 task tracking** | Read fully |
| `user/murilo/notes/2026-04-13-claude-deep-research-birdclef-strategy.md` | **Claude deep research** — unbiased BirdCLEF 2026 strategy analysis | Read fully |
| `user/murilo/notes/2026-04-13-gemini-deep-research-birdclef-strategy.md` | **Gemini deep research** — unbiased BirdCLEF 2026 strategy analysis | Read fully |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | **Competitor insights** — Boredom 0.938, Salman 0.922, techniques shared | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006–027) | Read fully |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference (13 sections) | Skim |
| `user/murilo/experiments/025-iter-pseudo-labeling/results.md` | Exp 025 Phase B+C results (both failed) | Skim |

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

Both pseudo-label approaches failed. Waveform mixing was worse than row-addition.

### Re-evaluation: Deep research completed

Key realization: the 0.906→0.93+ gap vs competitors (Boredom 0.938, Salman 0.922)
using the **same labeled data** is the real problem. Ran unbiased deep research with
both Claude and Gemini (no context of our experiments).

**Convergent findings from both:**
- The gap is about **domain adaptation**, not architecture
- **Environmental noise augmentation** is critical (different from bird+bird MixUp)
- **Secondary labels** from focal data are underutilized (multi-species supervision)
- **Long-tail/non-bird taxa** (Insecta, Amphibia) are systematically undertreated
- **Post-processing** (TopN temporal consistency) is a free +0.01-0.015
- **SpecAugment** and **balanced sampling** are standard techniques we're missing

### Exp 028 proposed: Domain Shift Augmentation

Created exp 028 with three phases:
- **Phase A**: Diagnostic — analyze v5j predictions on labeled soundscapes (existing
  inference data at `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/`). Per-species
  and per-taxonomy AUC, check "Invisible 28" (soundscape-only species). Zero compute.
- **Phase B**: TopN=1 post-processing — inference-only kernel change. Zero training cost.
- **Phase C**: Combined training recipe — environmental noise mixing, secondary labels,
  soundscape oversampling, SpecAugment, sqrt-balanced sampling. All five changes
  tested together (21 individual changes have all failed — hypothesis: complementary).

**Running total: 21+ changes, 0 LB improvements.**

## What needs to happen next

### Priority 1: Start exp 028 Phase A — Diagnostic analysis

Use existing v5j inference data on labeled soundscapes to understand WHERE we fail:
- v5j logits: `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/train_soundscapes/logits.parquet`
- 4-fold logits: `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/`
- Ground truth: `shared/data/birdclef-2026/train_soundscapes_labels.csv`
- Adapt script from `025-iter-pseudo-labeling/diagnose_regression.py`

Key questions:
1. Per-species AUC on labeled soundscapes — which species are we worst at?
2. Per-taxonomy breakdown — Aves vs Amphibia vs Insecta performance
3. "Invisible 28" (soundscape-only: 25 insect sonotypes + 3 amphibians) — scoring 0.0?
4. Confidence distributions — focal-trained vs soundscape-only species

### Priority 2: Start exp 028 Phase B — TopN post-processing

Implement in submission kernel (can run in parallel with Phase A):
```python
prob[file, segment, class] *= max(prob[file, :, class])
```
Apply to existing v5j ONNX model, submit to LB. Zero training cost.

### Priority 3: Exp 028 Phase C — Combined training recipe

Based on Phase A findings, implement and train the combined recipe.

## Key patterns (updated 2026-04-13)

- **21+ changes, 0 LB successes** — includes 19 training recipe changes AND 2
  pseudo-label approaches. Neither recipe changes nor data additions improved LB.
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.922) beat us by
  0.016-0.032 using the same focal + labeled soundscape data. No KD, no Perch.
- **Domain shift is the core problem**: Both deep research analyses converge on this.
  Our model trains on clean focal recordings but is tested on noisy soundscapes.
  We're undertreating: noise augmentation, secondary labels, non-bird taxa,
  long-tail species, and post-processing.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Pseudo-labels
  are dead (2 approaches). Mel norm, max trick, multi-context all regressed.
- **Offline metrics remain anti-correlated with LB** — DO NOT trust them.
- **Individual changes fail; combined approach untested** — every single change
  regressed. The hypothesis is these improvements are complementary.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/028-domain-shift-augmentation/` | **Active experiment — start here** |
| `user/murilo/experiments/028-domain-shift-augmentation/proposal.md` | Experiment proposal (3 phases) |
| `user/murilo/experiments/028-domain-shift-augmentation/tasks.md` | Task tracking |
| `user/murilo/notes/2026-04-13-claude-deep-research-birdclef-strategy.md` | Claude's strategy analysis |
| `user/murilo/notes/2026-04-13-gemini-deep-research-birdclef-strategy.md` | Gemini's strategy analysis |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights |
| `user/murilo/experiments/025-iter-pseudo-labeling/results.md` | Exp 025 results (both phases failed) |
| `user/murilo/experiments/025-iter-pseudo-labeling/diagnose_regression.py` | Diagnostic script to adapt for Phase A |
| `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/` | **Existing v5j inference** (focal + soundscapes) |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference |
| `user/murilo/FINDINGS.md` | What worked / didn't across all experiments |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints |
