# Experiment 023: Perch + SED Ensemble — Results

## All Submissions

| Variant | W_PERCH | LB | Runtime (dry) | Notes |
|---------|---------|-----|---------------|-------|
| **Phase A (w=0.5)** | **0.50** | **0.927** | **145s (20 files)** | **New best overall. +0.002 vs Perch TF (0.925)** |
| Phase B (w=0.3) | 0.30 | — | — | Pending |
| Phase B (w=0.7) | 0.70 | — | — | Pending |

## Phase A Analysis (w=0.5, 0.927 LB)

### Score Breakdown

| Component | LB (standalone) | Notes |
|-----------|----------------|-------|
| Perch ProtoSSM v18 (TF) | 0.925 | Best individual model |
| Perch ProtoSSM v18 (ONNX) | 0.920 | −0.005 ONNX penalty |
| SED v5j-4fold-pp2 | 0.917 | Best SED |
| **Ensemble (w=0.5)** | **0.927** | **+0.002 vs best individual, +0.007 vs ONNX Perch** |

### Interpretation

The ensemble at 0.927 beats the best individual model (Perch TF 0.925) despite
using the weaker ONNX Perch backend (0.920). This confirms the models are
complementary — the +0.007 gain over ONNX Perch more than compensates for the
−0.005 ONNX penalty.

The gain is modest (+0.002 over Perch TF), suggesting significant overlap in
what both models get right. The weight grid (Phase B) will reveal whether
the optimal blend favors Perch or is truly balanced.

### Timing (Dry-Run, 20 Train Files)

| Component | Time | Est. 600 files |
|-----------|------|----------------|
| Perch ONNX load | 5.2s | 5s |
| Perch ONNX inference | 33s | ~10 min |
| ProtoSSM + MLP + PP | 9s | ~8 min |
| SED 4-fold + TTA | 103s | ~46 min |
| **Total** | **145s** | **~64 min** |

Well within 90-min Kaggle CPU limit.
