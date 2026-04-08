# Experiment 023: Perch + SED Ensemble — Tasks

## Goal

Combine Perch ProtoSSM v18 (0.925 LB) with SED v5j-4fold-pp2 (0.917 LB) in a
single Kaggle kernel. Target: 0.930+ LB.

## Reference Chain

| Model | LB | Notes |
|-------|-----|-------|
| Perch ProtoSSM v18 (TF) | 0.925 | Best overall (exp 020) |
| Perch ProtoSSM v18 (ONNX) | 0.920 | −0.005 ONNX penalty |
| SED v5j-4fold-pp2 | 0.917 | Best SED (exp 021) |
| Competition #1 | 0.939 | Target |

Status: **Phase A complete: 0.927 LB (new best overall).** Phase B submitted, awaiting LB scores.

**Branch**: `mg/023-perch-sed-ensemble` — PR #25

---

## Phase A: Simple Ensemble (w=0.5)

Build kernel, validate, submit.

- [x] A.1 Create experiment directory + proposal
- [x] A.2 Build `submission_ensemble.py` combining both pipelines (1,977 lines)
  - Perch ONNX inference (replace TF with ONNX: `onnx_out[0]`=emb, `onnx_out[3]`=logits)
  - ProtoSSM + MLP probes + 6-stage PP (from pretrained artifacts)
  - SED 4-fold ONNX inference + TTA + smoothing + TopN + priors
  - Weighted average (w=0.5)
- [x] A.3 Create `kernel-metadata-ensemble.json` with 5 dataset deps + model_sources
- [x] A.4 Syntax verification (local) — 54 functions, 4 classes, all checks passed
- [x] A.5 Push to Kaggle, dry-run success (20 train files, 145s)
  - Perch: 42s (ONNX load 5.2s, inference 33s, ProtoSSM+PP 9s)
  - SED: 103s (4-fold × 20 files, first file 15s cold start, rest ~4-5s)
  - Estimated ~64 min for ~600 test files (within 90 min limit)
- [x] A.6 Submit to LB → **0.927 LB** (new best overall, +0.002 vs Perch TF 0.925)

## Phase B: Weight Grid Search (2 more submissions)

Config-driven submission system added: YAML configs in `configs/`, `push.sh <config_name>`
injects override into kernel, creates separate Kaggle notebook per config.

- [x] B.0 Add config-driven push system (YAML configs + push.sh injection)
- [ ] B.1 Submit w=0.3 → dry-run passed, awaiting LB score
  - Kernel: `birdclef-2026-perch-sed-w03`
- [ ] B.2 Submit w=0.7 → dry-run passed, awaiting LB score
  - Kernel: `birdclef-2026-perch-sed-w07`
- [ ] B.3 Pick optimal weight, document findings

## Phase C: Advanced Ensembling (if needed)

- [ ] C.1 Per-taxa weighting (different w for Aves vs texture taxa)
- [ ] C.2 Calibration: normalize probability scales before averaging
- [ ] C.3 PP order: ensemble before PP vs after PP

---

## Results Tracking

| Variant | W_PERCH | LB | Runtime | Notes |
|---------|---------|-----|---------|-------|
| **Phase A (w=0.5)** | **0.50** | **0.927** | **~64 min est.** | **New best overall** |
| Phase B (w=0.3) | 0.30 | — | — | SED-heavy, awaiting LB |
| Phase B (w=0.7) | 0.70 | — | — | Perch-heavy, awaiting LB |
