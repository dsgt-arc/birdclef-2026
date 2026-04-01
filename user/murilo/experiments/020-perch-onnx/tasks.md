# Experiment 020: Perch ONNX + ProtoSSM Improvements — Tasks

## Goal

Fix Perch v4.2 regression, swap TF→ONNX for 9x speedup, scale ProtoSSM
toward 0.925-0.930, then ensemble Perch + SED within 90-min kernel budget.

**Current Perch Best**: 0.918 LB (ProtoSSM v4, exp 018)
**Current SED Best**: 0.913 LB (v5j-4fold, exp 019)
**Target**: 0.925-0.930 Perch alone, 0.930+ ensemble

### Reference Chain

| Score | Model | Key technique |
|-------|-------|---------------|
| 0.899 | Our Perch v4.2 | REGRESSED — bisect needed |
| 0.913 | Our SED v5j-4fold | HGNetV2-B0, 4-fold, NNAudio mel |
| 0.918 | Our Perch v4 | ProtoSSM d=128, 3-fold OOF |
| 0.925 | Yuriy ProtoSSM v5 | d=256, 3 layers, 6-stage PP |
| 0.930 | pantanal-distill V18 | d=320, 4 layers, MLP-only, per-class thresholds |
| 0.939 | Competition #1 | Unknown |

### Status: **Proposed** — starting Phase A

- **Branch**: `mg/020-perch-onnx`

---

## Phase A: Recover 0.918 + Bisect v4.2 Regression (1-2 days, 5 LB subs)

v4.2 applied 4 changes simultaneously, LB dropped 0.918→0.899. Need to bisect.

v4.2 changes:
1. Taxonomy aux head: genus (161 groups) → class_name (5 groups) — PRIME SUSPECT
2. Site mapping: 0-based → 1-based indexing — SECOND SUSPECT
3. MLP probes: added n_iter_no_change=15, random_state=42 — likely minor
4. Ensemble weight: OOF-optimized → fixed 0.5 — medium risk

- [ ] A.0 Reconstruct v4 baseline by reverting 4 changes in current v4.2 kernel.
      Also pull v4 from Kaggle (`kaggle kernels pull`) for diffing.
- [ ] A.1 Submit reconstructed v4 to LB — must score 0.918
- [ ] A.2 Bisect: taxonomy change only (genus → class_name)
- [ ] A.3 Bisect: site mapping only (0-based → 1-based)
- [ ] A.4 Bisect: MLP params only (n_iter_no_change, random_state)
- [ ] A.5 Bisect: ensemble weight only (OOF → fixed 0.5)
- [ ] A.6 Decision: identify culprit(s), document findings

---

## Phase B: Swap TF → ONNX Perch (1 day, 1 LB sub)

Replace TensorFlow Perch inference with ONNX Runtime for ~9x speedup.
Source: `justinchuby/Perch-onnx` on HuggingFace (Apache 2.0).

- [ ] B.1 Validate ONNX locally on PACE — numerical equivalence (max diff < 2e-5)
- [ ] B.2 Upload Perch ONNX model to Kaggle as dataset (`birdclef-2026-perch-onnx`)
- [ ] B.3 Modify kernel: remove TF, replace with onnxruntime session
      - Remove TF wheel install + imports
      - Replace `load_perch_model()` → `load_perch_onnx()`
      - Replace `run_perch_inference()` → ONNX `session.run()`
      - Reuse onnxruntime pattern from SED kernel (exp 010)
- [ ] B.4 Submit ONNX kernel to LB — must score 0.918, runtime < 15 min

---

## Phase C: Pantanal-Distill Improvements (3-5 days, 6-7 LB subs)

Each step applies ONE change. LB-validated before proceeding. Revert if regression.
Reference: `user/murilo/notes/pantanal-distill-0930-annotated.md`

- [ ] C.1 OOF-optimized ensemble weight (use `best_w` from grid search)
- [ ] C.2 5-fold OOF (from 3-fold): `oof_n_splits: 3→5`
- [ ] C.3 Per-class threshold optimization: 10-value OOF grid per species
- [ ] C.4 Scaled probes: PCA 64→128, MLP (128,)→(256,128), max_iter 300→500
- [ ] C.5 Scale ProtoSSM: d_model 128→256, n_layers 2→3
- [ ] C.6 Training improvements: focal_gamma 2.0→2.5, epochs 60→80, lr 1e-3→8e-4,
      label_smoothing 0.02→0.03, cosine restart scheduler (T=20)
- [ ] C.7 Full V18 scale: d_model 256→320, n_layers 3→4, cross_attn_heads 4→8,
      ResidualSSM d=64→128, layers 1→2, TTA [0,±1]→[0,±1,±2]

---

## Phase D: Perch + SED Ensemble (2-3 days, 2 LB subs)

After Phase C achieves ≥0.920 Perch score. Enabled by Phase B ONNX speedup.

Runtime budget: ~7 min (Perch ONNX) + ~48 min (SED 4-fold) + ~3 min (rest) = ~58 min

- [ ] D.1 Create unified kernel: Perch ProtoSSM + SED v5j-4fold
      - Audio loaded once, shared between Perch (raw) and SED (mel)
      - Both produce per-(file, window) probabilities
      - Ensemble: `w * perch + (1-w) * sed`
- [ ] D.2 Determine ensemble weight (start fixed 0.5, optimize later)
- [ ] D.3 Submit ensemble to LB — target ≥0.925

---

## Results Tracking

| Submission | LB | Runtime | Notes |
|------------|-----|---------|-------|
| Perch v4 (exp 018) | 0.918 | ~82 min | TF baseline |
| Perch v4.2 (exp 018) | 0.899 | ~82 min | REGRESSED |
| SED v5j-4fold (exp 019) | 0.913 | ~50 min | Best SED |
