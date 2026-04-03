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

### Status: **Exact .ipynb reproduction submitted, awaiting LB score. Competition #1 = 0.942.**

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
- [x] A.5 v3 push (with jaejohn/perch-meta cache): submitted.
- [x] A.6 **Both v2 and v3 TIMEOUT on LB submission.**

### Root Cause: V18 Config Override Causes Timeout

The notebook has two config layers:
- **V17 CFG** (Cell 5): d_model=256, 3 layers, 40 epochs, 3 TTA shifts
- **V18 CFG** (Cell 6): d_model=320, 4 layers, 80 epochs, 5 TTA shifts

Cell 6 **always executes**, overriding V17 with V18. The V18 config is much
heavier (bigger model, 2x epochs, 5 TTA shifts). This pushes the kernel over
the 90-minute Kaggle budget.

Additionally, our TF install was missing `--no-deps`, installing full dependency
chains instead of just the two specific wheels.

- [x] A.7 Clean V17 rewrite: fresh extraction from Kaggle pull (2026-04-02),
      V18 override removed, TF install fixed with `--no-deps`,
      V18 hardcoded thresholds replaced with V17 defaults (0.5).
      Verified: all 43 functions/classes identical to original.
      See `pantanal-implementation.md` for full step-by-step analysis.
- [x] A.8 Push v4 (V17 clean) to Kaggle and submit to LB.
      Dry-run: 230s (down from timeout). Cache loaded, ProtoSSM 23ep (early stop),
      ResidualSSM ran (wall=0.4min), all 6 PP stages applied.
- [x] A.9 LB result: **0.922** (+0.004 over our v4 0.918, but −0.008 vs pantanal 0.930).
      Gap analysis: our kernel uses jaejohn/perch-meta cache (may differ from
      pantanal's own extraction), V17 default thresholds (0.5) vs V18 hardcoded,
      and ENSEMBLE_WEIGHT_PROTO=0.5 (pantanal train mode found 0.0 = MLP-only).
      See results discussion below.

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
- [x] B.2 Upload Perch ONNX model to Kaggle as dataset (`birdclef-2026-perch-onnx`)
      Includes: perch_v2.onnx (391MB), labels.csv, onnxruntime-1.24.4 wheel (17MB)
- [ ] B.3 Create our own perch-meta dataset (replace jaejohn dependency)
      Script ready: `scripts/build_perch_meta.py` (supports --use-onnx flag)
- [x] B.4 Modify kernel: remove TF, replace with onnxruntime session
      - Removed TF wheel install + all TF imports
      - ONNX session loaded from bundled dataset
      - onnxruntime installed from bundled wheel (--no-deps, offline)
      - Also changed ENSEMBLE_WEIGHT_PROTO 0.5→0.0 (MLP-only)
- [x] B.5 v5 pushed (nb v6): dry-run 98s (was 230s with TF), 2.3x faster.
      LB: 0.911 (MLP-only w=0.0 regressed). Runtime: 20 min (great speedup).
- [x] B.6 v7 (nb v7): V18+ONNX, w=0.5, V18 thresholds. LB: **0.920**.
      ONNX causes ~0.010 LB penalty vs TF (0.930→0.920).
      Penalty compounds through probe features, gated fusion, V18 thresholds,
      and 5 TTA shifts.

### ONNX Finding: −0.010 LB penalty

ONNX is numerically close (max 2.7e-4 logit diff) but the downstream pipeline
amplifies this. The V18 per-class thresholds were optimized on TF outputs and
are mismatched with ONNX. Phase B is **on hold** — reverting to TF for 0.930.

### Phase B Revised Plan

- [x] B.7 V18+TF as .py script: v8 (nb v8) timed out on LB, v9 (nb v9) failed
      (TF 2.19 loaded, ashok205 path wrong), v10 (nb v10) compiled in 464s but
      TF inference still 10.9s/file — .py script format too slow for V18+TF.
- [x] B.8 Discovery: .ipynb runs TF at 5.8s/file (2x faster than .py's 10.9s/file).
      Root cause unknown — same code, different Kaggle runtime behavior.
- [x] B.9 Exact .ipynb reproduction: pushed unmodified pantanal notebook as
      `birdclef-2026-perch-protossm-v5-nb`. Dry-run 666s (cache not found,
      extracted fresh 337s). **Running LB inference — awaiting score.**
- [ ] B.10 Check LB score. If 0.930 confirmed, test ONNX as isolated variable.
- [ ] B.11 If ONNX penalty confirmed, explore: retune thresholds on ONNX outputs,
      or accept penalty and use ONNX only for Perch+SED ensemble (Phase D)

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
| ProtoSSM v5 port v2 (V18, no cache) | timeout | 1110s dry | V18 override caused timeout |
| ProtoSSM v5 port v3 (V18, cache) | timeout | — | V18 override caused timeout |
| ProtoSSM v5 port v4 (V17, cache) | **0.922** | 230s dry | V17 clean rewrite, ResidualSSM ran, new best Perch |
| ProtoSSM v5 port v5 (ONNX+MLP-only) | **0.911** | 98s dry, 20 min LB | TF→ONNX works, but MLP-only regressed −0.011 |
| ProtoSSM v5 port v7 (V18+ONNX) | **0.920** | 242s dry | V18 config + ONNX. ONNX penalty −0.010 vs TF |
| ProtoSSM v5 v8 (.py V18+TF) | timeout | 464s dry | .py script TF inference 10.9s/file, too slow |
| ProtoSSM v5 v9 (.py V18+TF) | error | 173s | TF 2.19 loaded (ashok205 path wrong) |
| ProtoSSM v5 v10 (.py V18+TF+gc) | timeout* | 464s dry | gc.collect didn't help, still 10.9s/file |
| ProtoSSM v5-nb v1 (.ipynb V18+TF) | **running** | 666s dry | Exact .ipynb repro, TF 5.8s/file, no cache |
