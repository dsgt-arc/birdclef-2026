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

Status: **Phases A+B+C complete. Best: 0.929 LB (MLP-only, new best overall).** ProtoSSM hurts in ONNX ensemble.

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
- [x] B.1 Submit w=0.3 → **0.927 LB** (tied with w=0.5)
  - Kernel: `birdclef-2026-perch-sed-w03`
- [x] B.2 Submit w=0.7 → **0.926 LB** (−0.001)
  - Kernel: `birdclef-2026-perch-sed-w07`
- [x] B.3 Weight curve flat (0.926-0.927). Both models contribute equally.
  ONNX penalty is the binding constraint, not the blend weight.

## Phase C: Ablations

- [x] C.1 MLP-only (proto_weight=0.0) → **0.929 LB** (new best! +0.002 vs ProtoSSM+MLP)
  - Kernel: `birdclef-2026-perch-sed-mlp_only`
  - Matches pantanal-distill 0.930 approach. ProtoSSM hurts under ONNX mismatch.
- [x] C.2 No PP (raw ensemble, both PPs disabled) → **0.922 LB** (−0.005 vs w=0.5)
  - Kernel: `birdclef-2026-perch-sed-no_pp`
  - Confirms PP is worth +0.005. Individual PPs help even when averaged.
- [ ] C.3 TF Perch + SED single-fold (recover ONNX penalty) — requires code changes

## Future Directions

- [ ] D.1 Per-taxa weighting (different w for Aves vs texture taxa)
- [ ] D.2 Calibration: normalize probability scales before averaging

---

## Results Tracking

| Variant | W_PERCH | LB | Runtime | Notes |
|---------|---------|-----|---------|-------|
| **Phase A (w=0.5)** | **0.50** | **0.927** | **~64 min est.** | **New best overall** |
| **Phase B (w=0.3)** | **0.30** | **0.927** | **~64 min est.** | **Tied with w=0.5** |
| Phase B (w=0.7) | 0.70 | 0.926 | ~64 min est. | −0.001 |
| **Phase C MLP-only** | **0.50** | **0.929** | **~64 min est.** | **New best! proto_weight=0.0** |
| Phase C No PP | 0.50 | 0.922 | ~64 min est. | Both PPs disabled, −0.005 |
