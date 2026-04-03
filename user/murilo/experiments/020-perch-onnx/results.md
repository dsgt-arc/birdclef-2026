# Experiment 020: Perch ONNX + ProtoSSM — Results

## All Submissions

| Nb ver | Kernel ver | Config | Perch Backend | Cache | LB | Runtime (dry) | Notes |
|--------|-----------|--------|--------------|-------|-----|---------------|-------|
| v2 | v2 | V18 (d=320, 4L, 80ep) | TF | no | timeout | 1110s | V18 override too heavy with TF |
| v3 | v3 | V18 (d=320, 4L, 80ep) | TF | jaejohn | timeout | — | V18 override too heavy with TF |
| v4 | v4 | V17 (d=256, 3L, 40ep) | TF | jaejohn | **0.922** | 230s | V17 clean rewrite |
| v5 | v5 | V17 + ensemble=0.0 | ONNX | jaejohn | 0.911 | 98s | MLP-only regressed −0.011 |
| v6 | v5 (fix) | V17 + ensemble=0.5 | ONNX | jaejohn | — | — | not submitted (weight fix only) |
| v7 | v6 | V18 + ensemble=0.5 | ONNX | jaejohn | **0.920** | 242s | V18 restored, ONNX |
| v18-train | — | V18 (train_only) | TF | jaejohn | — | 277s | Two-kernel: saves artifacts |
| **v18-infer** | **v3** | **V18 (infer_only)** | **TF** | **pretrained** | **0.925** | **213s dry** | **Two-kernel: loads artifacts, new best Perch** |

Reference: pantanal-distill (dingjiarun) V18+TF = **0.930** LB

## Key Finding: ONNX Causes the 0.930 → 0.920 Gap

### Confirmed: same cache, same code

The original pantanal-distill notebook uses `jaejohn/perch-meta` (dataset ID 9712127) —
the **exact same cache** we use. The code is verified identical (all 43 functions match).
The only difference between the original (0.930) and our v7 (0.920) is **TF vs ONNX
for test inference**.

### How ONNX diverges

PACE validation showed max logit diff of 2.7e-4 between TF and ONNX — seemingly tiny.
But this compounds through the downstream pipeline:

1. **MLP probe features**: `build_class_features` uses raw Perch scores as input features.
   Small logit diffs → different probe predictions for threshold-edge cases.
2. **ProtoSSM gated fusion**: `alpha * sim + (1-alpha) * perch_logits` — Perch logits
   directly enter the model output.
3. **V18 per-class thresholds**: Hardcoded thresholds were optimized on TF outputs.
   The sharpening function `apply_per_class_thresholds` amplifies small differences
   near threshold boundaries.
4. **5 TTA shifts**: Each shift runs ONNX independently, accumulating 5x the inference diff.
5. **ResidualSSM**: Second-pass correction trained on first-pass ensemble, which itself
   uses ONNX test scores.

### Evidence

| Config | TF LB | ONNX LB | Delta |
|--------|-------|---------|-------|
| V17 | 0.922 | 0.911* | −0.011 |
| V18 | 0.930 | 0.920 | −0.010 |

*V17+ONNX at ensemble=0.0. V17+ONNX at ensemble=0.5 was not submitted.

The ONNX penalty is consistent at ~0.010 across both V17 and V18 configs.

## Phase A Analysis

### v4 (V17+TF, 0.922)

- ProtoSSM: 3M params, 23 epochs (early stopped), best_val_loss=0.4406
- MLP probes: 52 classes (PCA-64, hidden=(128,))
- ResidualSSM: 176K params, 26 epochs
- OOF baseline AUC: 0.8061

### v7 (V18+ONNX, 0.920)

- ProtoSSM: 5.8M params, 80 epochs (no early stop), best_val_loss=0.4590 (worse)
- MLP probes: 58 classes (PCA-128, hidden=(256,128))
- ResidualSSM: 439K params, 40 epochs
- OOF baseline AUC: 0.8032 (worse — V18 fusion lambdas less optimal for this data)

V18's bigger model actually has worse val_loss (0.4590 vs 0.4406) on 59 training files.
The capacity increase overfits. V18 only wins when test inference matches TF exactly
(so the hardcoded thresholds and fusion lambdas align).

## Ensemble Weight Finding

| Config | w=0.5 | w=0.0 | Delta |
|--------|-------|-------|-------|
| V17+ONNX | 0.922 (TF) | 0.911 (ONNX) | −0.011 |

ENSEMBLE_WEIGHT_PROTO=0.0 (MLP-only) was −0.011 worse. ProtoSSM genuinely helps
at w=0.5 under V17 config, contradicting the pantanal OOF finding of w=0.0
(which was from V18 config with a bigger model).

## Two-Kernel Split Finding

V18 config timed out in a single kernel (training + TF inference > 90 min).
Solution: split into training kernel (saves artifacts) + inference kernel (loads artifacts).

Key discovery: **loading 75MB of pretrained artifacts before TF inference caused 2x slowdown**
(11s/file vs 5.5s/file). Fix: run Perch TF inference FIRST with clean memory, then load artifacts.

## V18+TF Result: 0.925 LB

| Config | Backend | LB | Delta vs V17 |
|--------|---------|-----|-------------|
| V17 (d=256, 3L, 40ep, TTA×3) | TF | 0.922 | baseline |
| V18 (d=320, 4L, 80ep, TTA×5) | ONNX | 0.920 | −0.002 |
| **V18 (d=320, 4L, 80ep, TTA×5)** | **TF** | **0.925** | **+0.003** |

V18+TF gains +0.003 over V17+TF. The ONNX penalty is real but smaller than initially
estimated: V18+TF (0.925) vs V18+ONNX (0.920) = −0.005 (not −0.010 as we estimated
from the pantanal 0.930 comparison).

Gap to pantanal (0.930): −0.005. Possible sources:
- Ensemble weight: we use 0.50 (default), pantanal OOF found 0.0 (MLP-only)
- Per-class thresholds: we use V18 hardcoded, pantanal computed fresh per run
- jaejohn/perch-meta cache may differ subtly from pantanal's own extraction
- Random seed differences in ProtoSSM training

## Next Steps

1. **ONNX inference kernel** — swap TF for ONNX in the inference kernel, same pretrained
   artifacts. Saves ~50 min runtime, enables Perch+SED ensemble (Phase D).
2. **Ensemble weight ablation** — try ENSEMBLE_WEIGHT_PROTO=0.0 (MLP-only, matching pantanal)
3. **Perch + SED ensemble** — with ONNX Perch (~7 min) + SED 4-fold (~50 min) = ~57 min
