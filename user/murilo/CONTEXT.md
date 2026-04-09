# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-08

## Read these files first

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns | Read fully — it's the most important doc |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim latest sections (exp 020-023) |
| `user/murilo/EXPERIMENTS.md` | Per-experiment methodology summaries (001-023) | Skim Section 1 table, read exp 019-023 details |
| `user/murilo/experiments/023-perch-sed-ensemble/results.md` | Current experiment results | Read fully |
| `user/murilo/experiments/023-perch-sed-ensemble/tasks.md` | Current experiment task tracking | Read fully |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Target: 0.939** (competition #1). Gap: 0.010.
**Branch**: `mg/023-perch-sed-ensemble` — PR #25

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold, proto_weight=0.0** |
| Perch+SED w=0.5 (exp 023) | 0.927 | ProtoSSM+MLP blend |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED |
| SED v5j single (exp 019) | 0.906 | Best single-fold SED |

## What happened in the last session

1. **Exp 023: Perch + SED Ensemble — 0.929 LB (new best overall)**
   - Built single Kaggle CPU kernel combining Perch ONNX + SED v5j-4fold-pp2 (~2,200 lines)
   - Config-driven submission system: YAML configs in `configs/`, `push.sh <config_name>`
   - Phase A: w=0.5 → 0.927 LB (+0.002 vs Perch TF alone)
   - Phase B: weight grid flat (w=0.3: 0.927, w=0.5: 0.927, w=0.7: 0.926)
   - Phase C: **MLP-only (proto_weight=0.0): 0.929 LB** — best result
   - Phase C: No PP: 0.922 (−0.005) — confirms PP adds value
   - Phase C: TF Perch + single SED **timed out** — TF ~10s/file exceeds 90 min for 600 files

2. **Documentation overhaul** — Three source-of-truth docs:
   - `EXPERIMENTS.md` — per-experiment methodology (updated through exp 023)
   - `MODEL-RUNS.md` — all model scores and checkpoints
   - `FINDINGS.md` — cross-cutting insights and patterns (new)

## What needs to happen next

### Resolved: TF+single timed out
- TF Perch + SED single-fold + MLP-only timed out (TF ~10s/file, >90 min)
- TF backend is not viable for ensemble kernels
- ONNX Perch (0.929 MLP-only) remains the best path

### High-value next steps (from FINDINGS.md gap analysis)

| Option | Effort | Expected gain | Risk |
|--------|--------|---------------|------|
| MLP-only weight grid (w=0.3, 0.7) | Zero | +0.001 | Low — weight curve was flat for ProtoSSM+MLP |
| Framewise max inference (Nikita 2025 trick) | Low | Unknown | Inference-only change, no retraining |
| Better ensemble PP (unified PP after merge) | Medium | +0.001-0.003 | No-PP scored 0.922, current PP adds +0.005 |
| Knowledge distillation from ensemble | High | +0.003-0.005 | Exp 015 showed pseudo-labels can hurt |
| SED embedding probes (OOF linear) | Medium | Unknown | Diagnostic — reveals complementarity |

## Key patterns (summary from FINDINGS.md)

- **Only input representation and inference-time changes improve LB** — 15+ training recipe changes, 0 successes
- **Offline metrics are anti-correlated with LB** — don't trust val/auroc
- **Simpler models beat complex ones in ensemble** — MLP-only > ProtoSSM+MLP
- **PP is worth +0.005** — both pipelines' PP contribute positively
- **Weight doesn't matter much** — flat curve across w=0.3/0.5/0.7
- **ONNX penalty is real but manageable** — −0.005 vs TF, compensated by ensemble gain

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/023-perch-sed-ensemble/kaggle_kernel/submission_ensemble.py` | Combined ensemble kernel |
| `user/murilo/experiments/023-perch-sed-ensemble/configs/*.yaml` | Submission configs |
| `user/murilo/experiments/023-perch-sed-ensemble/kaggle_kernel/push.sh` | Config-driven Kaggle push |
| `user/murilo/experiments/020-perch-onnx/kaggle_kernel/submission_perch_protossm.py` | Perch ProtoSSM source (3,748 lines) |
| `user/murilo/experiments/010-submission-pipeline/kaggle_kernel/submission.py` | SED submission source (1,454 lines) |
| `user/murilo/inference/infer.py` | Offline inference pipeline (SED + ProtoCLR) |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor intel |
| `user/murilo/notes/2026-04-05-sed-gap-analysis.md` | SED gap analysis vs competitors |
