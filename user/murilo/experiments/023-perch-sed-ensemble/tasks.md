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

Status: Phase A implementation complete. Ready to push to Kaggle.

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
- [ ] A.5 Push to Kaggle, check logs for dry-run success
- [ ] A.6 Submit to LB → record score and runtime

## Phase B: Weight Grid Search (2 more submissions)

- [ ] B.1 Submit w=0.3 → record score
- [ ] B.2 Submit w=0.7 → record score
- [ ] B.3 Pick optimal weight, document findings

## Phase C: Advanced Ensembling (if needed)

- [ ] C.1 Per-taxa weighting (different w for Aves vs texture taxa)
- [ ] C.2 Calibration: normalize probability scales before averaging
- [ ] C.3 PP order: ensemble before PP vs after PP

---

## Results Tracking

| Variant | W_PERCH | LB | Runtime | Notes |
|---------|---------|-----|---------|-------|
| Phase A (w=0.5) | 0.50 | — | — | |
| Phase B (w=0.3) | 0.30 | — | — | |
| Phase B (w=0.7) | 0.70 | — | — | |
