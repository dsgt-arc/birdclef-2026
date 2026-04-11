# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-11

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/experiments/025-iter-pseudo-labeling/proposal.md` | **Current experiment** — full methodology (12 sections) with step-by-step walkthrough | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/tasks.md` | Current task tracking (Phase A ✅, Phase B in progress) | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/findings/v1-pseudo-labels.md` | v1 pseudo-label generation results and quality validation | Read fully |
| `user/murilo/experiments/025-iter-pseudo-labeling/README.md` | Experiment overview, directory structure, commands | Read fully |
| `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` | 14 sections of model analysis + Perch vs SED comparison | Skim sections 12-14 (Phase C/D/E findings) |
| `user/murilo/experiments/024-data-centric-exploration/results.md` | Exp 024 findings that motivate exp 025 | Skim synthesis section |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006-024) | Read fully |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim latest sections (exp 020-023) |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/025-iter-pseudo-labeling` — PR #27 (exp 024 merged to main via PR #26)

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold, proto_weight=0.0** |
| Perch+SED w=0.5 (exp 023) | 0.927 | ProtoSSM+MLP blend |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED |
| SED v5j single (exp 019) | 0.906 | Best single-fold SED — **baseline for exp 025** |

## What happened in this session (2026-04-11)

### 1. Exp 025 Setup

- Renamed experiment from `025-pseudo-labeling-v2` to `025-iter-pseudo-labeling`
- Created branch `mg/025-iter-pseudo-labeling`, opened PR #27
- Comprehensive proposal (12 sections), README, versioned configs, pyproject.toml

### 2. Phase A: v1 Pseudo-Labels Generated ✅

Script: `generate_pseudo_labels.py` with `configs/v1.yaml`
Output: `shared/pseudo-labels/birdclef-2026/v1/`

| Teacher | Tier | Species | Pseudo-labels >0.5 |
|---------|------|---------|---------------------|
| SED max-folds (T=0.6, P^1.5) | A+B | 162/184 | 198K |
| Perch z-norm (z>2.0, P^1.5) | C | 34/43 | 149K |
| **Merged** | **All** | **196/234** | **347K** |

Key validation: Perch z-normalization pairwise overlap = 5.1% (species-specific signal, not noise).
Mean 2.73 species/segment. Training balance: 19.3% PL / 80.7% GT.

### 3. Phase B: Training Pipeline Implemented ✅

Shared src modifications (backward compatible with exp 021):
- `src/birdclef/data/sed.py`: `PseudoLabelSoundscapeDataset`, `GTDatasetWrapper`,
  `StratifiedBatchSampler`, extended `SEDDataModule` with pseudo-label params
- `src/birdclef/lightning/sed_finetune.py`: per-sample loss weighting (BCE path),
  4-tuple batch `(x, labels, primary_idx, loss_weight)`, mixup loss_weight blending

Exp 025 files: `train.py`, `configs/v1_train.yaml`, `sbatch/`, `fold_assignments.json`

### 4. Validation

- **Fast-dev-run** ✅ — 5 batches, 32724 GT + 127896 PL = 160620 total,
  50/50 batch composition confirmed, loss_weight=0.5 for PL
- **GPU-check** ✅ — 1 epoch, memory and throughput OK on RTX 6000
- **Full training run submitted** — 90/10 split (same as v5j baseline), awaiting completion

## What needs to happen next

### Immediate — awaiting training completion

1. Check training logs when job completes (loss curves, val/auroc)
2. Export checkpoint → ONNX → Kaggle submission
3. Submit to LB → compare against v5j baseline (0.906)
4. **Decision gate**: If LB > 0.906 → proceed to 4-fold + ensemble

### If round 1 succeeds

5. Train 4-fold ensemble → submit (compare vs 0.913)
6. Student + Perch ensemble → submit (compare vs 0.929)
7. Student-as-teacher → v2 pseudo-labels → round 2

### If round 1 fails

5. Diagnose: check per-species analysis, val curves
6. Try variants: PL weight 0.3, power P^2.0, Perch z=2.5
7. Try without Tier C (SED-only pseudo-labels)

## Key patterns (updated)

- **Only data and inference changes improve LB** — 15+ training recipe changes, 0 successes
- **The LB gap is data coverage** — 0.9958 AUC on 75 known species, 159 species unseen
- **SED is precise but narrow** (94% precision, 109/159), **Perch is broad but noisy** (6%, 149/159)
- **Perch raw probs NOT directly usable** — 15M FPs, 119 species/segment
- **Per-species z-normalization works** — 5.1% overlap, extracts species-specific signal from Perch
- **v5j recipe unchanged** — only data changes. BCE loss, soft labels, w=0.5 for PL

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/025-iter-pseudo-labeling/` | Current experiment (proposal, tasks, configs, train.py) |
| `user/murilo/experiments/025-iter-pseudo-labeling/findings/v1-pseudo-labels.md` | v1 generation results |
| `user/murilo/src/birdclef/data/sed.py` | Data module with PL support |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | Training module with loss weighting |
| `shared/pseudo-labels/birdclef-2026/v1/` | v1 pseudo-labels (parquets + stats) |
| `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` | EDA findings (14 sections) |
| `user/murilo/experiments/024-data-centric-exploration/results.md` | Exp 024 synthesis |
| `user/murilo/experiments/023-perch-sed-ensemble/kaggle_kernel/submission_ensemble.py` | Submission pipeline reference |
