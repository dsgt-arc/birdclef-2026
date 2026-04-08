# Experiment 023: Perch + SED Ensemble — Results

## All Submissions

| Variant | W_PERCH | LB | Runtime (dry) | Notes |
|---------|---------|-----|---------------|-------|
| **Phase A (w=0.5)** | **0.50** | **0.927** | **145s (20 files)** | **New best overall. +0.002 vs Perch TF (0.925)** |
| **Phase B (w=0.3)** | **0.30** | **0.927** | **181s (20 files)** | **Tied with w=0.5** |
| Phase B (w=0.7) | 0.70 | 0.926 | 189s (20 files) | −0.001 vs w=0.5 |

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

## Phase B Analysis (Weight Grid)

| W_PERCH | LB | Delta vs w=0.5 |
|---------|-----|----------------|
| 0.30 | **0.927** | 0.000 |
| **0.50** | **0.927** | baseline |
| 0.70 | 0.926 | −0.001 |

The weight curve is remarkably flat — all three weights land at 0.926-0.927.
This suggests:

1. **Both models contribute roughly equally.** There's no clear winner; the
   ensemble benefits from both at any reasonable blend.
2. **The slight edge to SED-heavy (w=0.3) or balanced (w=0.5)** suggests SED
   adds marginally more unique signal than ONNX Perch in this ensemble.
3. **The ONNX penalty is the binding constraint.** If Perch were running at its
   TF quality (0.925 vs 0.920), w=0.7 would likely match or beat w=0.5.
4. **Diminishing returns from weight tuning.** The +0.001 difference between
   best (0.927) and worst (0.926) is within noise. Further weight fine-tuning
   (e.g., w=0.4, 0.6) is unlikely to yield meaningful gains.
