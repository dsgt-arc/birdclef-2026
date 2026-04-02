# Experiment 020: Perch ONNX + ProtoSSM Improvements — Tasks

## Goal

Port pantanal-distill V17 (0.930 LB) as new Perch baseline, swap TF→ONNX for
faster inference, then ensemble Perch + SED within 90-min kernel budget.

**Current Perch Best**: 0.918 LB (ProtoSSM v4, exp 018)
**Current SED Best**: 0.913 LB (v5j-4fold, exp 019)
**Target**: 0.925-0.930 Perch alone, 0.935+ ensemble

### Reference Chain

| Score | Model | Key technique |
|-------|-------|---------------|
| 0.913 | Our SED v5j-4fold | HGNetV2-B0, 4-fold, NNAudio mel |
| 0.918 | Our Perch v4 | ProtoSSM d=128, 3-fold OOF |
| 0.925 | Yuriy ProtoSSM v5 | d=256, 3 layers, 6-stage PP |
| 0.930 | pantanal-distill V17 | d=256, 3 layers, MLP-only, per-class thresholds |
| 0.939 | Competition #1 | Unknown |

### Status: **Phase A running, Phase B validation complete**

- **Branch**: `mg/020-perch-onnx`

---

## Phase A: Port Pantanal-Distill V17 as Baseline (1 day, 1-2 LB subs)

Directly port the pantanal-distill notebook (0.930 LB) as our new Perch baseline.
The code is self-contained (3418 lines) and already validated on LB.

Source: `user/murilo/notes/pantanal-distill-0930.py`
Annotated: `user/murilo/notes/pantanal-distill-0930-annotated.md`
Config: V17 (d=256, 3 layers) — the actual code, despite the file being labeled V18.

Kaggle datasets required:
- `birdclef-2026` (competition data)
- `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` (Perch model)
- `murilogustineli/birdclef-tf220-wheel` (TF 2.20 wheels)
- `jaejohn/perch-meta` (pre-extracted labeled embeddings cache, saves ~10 min)

- [x] A.1 Copy pantanal-distill code as kernel, add TF 2.20 wheel installer
      (matching exp 018 proven pattern)
- [x] A.2 Create kernel metadata JSON + push.sh entry
- [x] A.3 v1 push: failed — TF 2.19 loaded (wheel install wasn't running). Fixed.
- [x] A.4 v2 push (no cache): succeeded in 1110s. Labeled extraction took ~625s.
      Dry-run on 20 test files. Submitted to LB — awaiting result.
      Note: ResidualSSM skipped (wall time safety cutoff at 12.9 min).
- [ ] A.5 v3 push (with jaejohn/perch-meta cache): **currently running on Kaggle**.
      Should skip labeled extraction (~10 min saved), have time for ResidualSSM.
- [ ] A.6 Record LB results from v2 and/or v3, update results tracking.

---

## Phase B: Swap TF → ONNX Perch (1 day, 1 LB sub)

Replace TensorFlow Perch inference with ONNX Runtime for faster inference.
Source: `justinchuby/Perch-onnx` on HuggingFace (Apache 2.0).

### ONNX Validation Results (completed on PACE)

| Backend | Time (12 samples) | Per sample | Speedup |
|---------|-------------------|------------|---------|
| TF 2.21 | 21.2s | 1769ms | baseline |
| ONNX standard | 5.2s | 436ms | **4.1x** |
| ONNX no_dft | 5.3s | 439ms | 4.0x |

Numerical equivalence (ONNX standard vs TF):
- Embeddings max diff: **8e-6** (PASS, tol=1e-3)
- Logits max diff: **2.7e-4** (PASS, tol=1e-3) — ~0.007% probability diff after sigmoid

ONNX models downloaded to: `/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-onnx/`

- [x] B.1 Validate ONNX locally on PACE — **PASS** (max diff 2.7e-4, 4.1x speedup)
- [ ] B.2 Upload Perch ONNX model to Kaggle as dataset (`birdclef-2026-perch-onnx`)
- [ ] B.3 Create our own perch-meta dataset (replace jaejohn dependency)
      Script ready: `scripts/build_perch_meta.py` (supports --use-onnx flag)
- [ ] B.4 Modify kernel: remove TF, replace with onnxruntime session
      - Remove TF wheel install + imports
      - Replace Perch TF inference with ONNX `session.run()`
      - Reuse onnxruntime pattern from SED kernel (exp 010)
- [ ] B.5 Submit ONNX kernel to LB — must match Phase A score, runtime < 20 min

---

## Phase C: Further Improvements (if needed)

Only if Phase A doesn't match 0.930, apply incremental changes from the
annotated reference. Each step LB-validated before proceeding.

Reference: `user/murilo/notes/pantanal-distill-0930-annotated.md`

- [ ] C.1 Investigate V18 vs V17 config discrepancy (code has d=256/3 layers,
      annotated doc says V18 has d=320/4 layers — may need to find V18 code)
- [ ] C.2 5-fold OOF (if V17 uses 3-fold): `oof_n_splits: 3→5`
- [ ] C.3 Scale ProtoSSM: d_model 256→320, n_layers 3→4 (V18 params)
- [ ] C.4 Training improvements: focal_gamma, epochs, lr, cosine restart

---

## Phase D: Perch + SED Ensemble (2-3 days, 2 LB subs)

After Phase B ONNX is validated. Enabled by ONNX Perch speedup.

Runtime budget with ONNX: ~20 min (Perch) + ~50 min (SED 4-fold) + ~5 min (rest) = ~75 min

- [ ] D.1 Create unified kernel: Perch ProtoSSM + SED v5j-4fold
      - Audio loaded once, shared between Perch (raw) and SED (mel)
      - Both produce per-(file, window) probabilities
      - Ensemble: `w * perch + (1-w) * sed`
- [ ] D.2 Determine ensemble weight (start fixed 0.5, optimize later)
- [ ] D.3 Submit ensemble to LB — target ≥0.935

---

## Results Tracking

| Submission | LB | Runtime | Notes |
|------------|-----|---------|-------|
| Perch v4 (exp 018) | 0.918 | ~82 min | Previous best Perch |
| SED v5j-4fold (exp 019) | 0.913 | ~50 min | Best SED |
| ProtoSSM v5 port v2 (no cache) | pending | 1110s | Dry-run 20 files, ResidualSSM skipped |
| ProtoSSM v5 port v3 (perch-meta) | running | — | With jaejohn/perch-meta cache |
