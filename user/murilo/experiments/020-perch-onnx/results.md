# Experiment 020: Perch ONNX + ProtoSSM — Results

## Phase A: Port Pantanal-Distill V17

### LB Results

| Version | Config | Cache | LB | Runtime (dry) | Notes |
|---------|--------|-------|-----|---------------|-------|
| v2 | V18 (d=320, 4L, 80ep) | no | timeout | 1110s | V18 override too heavy |
| v3 | V18 (d=320, 4L, 80ep) | yes (jaejohn) | timeout | — | V18 override too heavy |
| **v4** | **V17 (d=256, 3L, 40ep)** | **yes (jaejohn)** | **0.922** | **230s** | Clean rewrite, V18 removed |

### v4 Key Metrics (from dry-run logs)

- ProtoSSM training: 23 epochs (early stopped at patience=8), 11.3s
- MLP probes trained: 52 classes
- ResidualSSM: ran (wall=0.4 min < 4 min cutoff), 26 epochs, 1.0s
- Ensemble weight: 0.50 (default, not OOF-optimized)
- Fusion alpha: mean=0.501 (ProtoSSM and Perch roughly equal)
- Correction magnitude: mean_abs=0.4601
- Final score range: 0.000 to 1.000, mean=0.221

### Gap Analysis: 0.922 vs 0.930 (pantanal reference)

Our v4 scores 0.922, which is +0.004 over our previous best (0.918, exp 018 v4)
but −0.008 below the pantanal-distill reference (0.930). Possible causes:

1. **perch-meta cache mismatch**: We use `jaejohn/perch-meta` which was extracted
   by a third party. The pantanal notebook may have used its own extraction with
   slightly different proxy_reduce settings or TF version. Perch embeddings are
   sensitive to TF version (we validated max diff 2.7e-4 between TF and ONNX).

2. **Per-class thresholds**: We use default 0.5 for all classes. The V18 pantanal
   has hardcoded per-class thresholds optimized from OOF. Even with V17 config,
   the original may have been using OOF-optimized thresholds from a train run.

3. **Ensemble weight**: We default to 0.50 (ProtoSSM=0.5, MLP=0.5). The pantanal
   train mode found ENSEMBLE_WEIGHT_PROTO=0.0 (MLP-only wins on OOF). Using 0.0
   instead of 0.5 could improve the score.

4. **Stochastic training**: ProtoSSM and MLP probes have random initialization.
   The pantanal's 0.930 may represent a lucky seed. Our early stopping at epoch 23
   (vs potentially running longer) may also matter.

5. **V17 vs actual pantanal version**: The pantanal notebook's 0.930 score may have
   been achieved with a version between V17 and V18 that we don't have — e.g., V17
   with per-class thresholds but without the heavy V18 model scaling.

### Recommended Next Steps

1. **Try ENSEMBLE_WEIGHT_PROTO=0.0** (MLP-only) — free, just change one constant
2. **Build our own perch-meta** — ensures cache matches our exact pipeline
3. **Swap TF→ONNX** (Phase B) — 4x speedup, frees budget for improvements
4. **Add per-class thresholds from OOF** — run train mode to get V17-specific thresholds
