---
experiment: "020-perch-onnx"
status: proposed
date: 2026-04-01
author: Murilo Gustineli
---

# Experiment 020: Perch ONNX + ProtoSSM Improvements

## Hypothesis

Replacing TensorFlow Perch inference with ONNX (~9x speedup) will unlock a
Perch+SED ensemble that exceeds either model alone, and applying pantanal-distill
V18 scaling improvements will push Perch ProtoSSM from 0.918 toward 0.925-0.930 LB.

## Background

### Current state

Two strong model families:
- **Perch ProtoSSM v4**: 0.918 LB — frozen Perch v2 + learned temporal model + MLP
  probes + 6-stage post-processing. Runs ~82 min on Kaggle CPU kernel.
- **SED v5j-4fold**: 0.913 LB — HGNetV2-B0 trained CNN, 4-fold ensemble. Runs ~50 min.

### Three problems

1. **v4.2 regression** (0.918→0.899): Applied 4 changes simultaneously in exp 018,
   LB dropped by 0.019. Need to bisect and recover baseline.

2. **Perch TF too slow**: ~80 min of the ~82 min total kernel time is Perch TF
   inference. Combined with SED (~50 min), ensemble would need ~130 min — over the
   90-min Kaggle budget. Ensemble is currently impossible.

3. **Room to improve Perch**: pantanal-distill V18 scores 0.930 with the same
   ProtoSSM architecture, just scaled up (d_model 128→320, 4 layers, PCA-128,
   5-fold OOF, per-class thresholds).

### Perch ONNX opportunity

A community-converted Perch ONNX model exists on HuggingFace
(`justinchuby/Perch-onnx`, Apache 2.0):
- **9.2x faster** than TFLite (66ms vs 609ms per sample)
- Numerically identical (max diff ~1.5e-5)
- Input: `[batch, 160000]` float32 → Output: embedding `[batch, 1536]` + label `[batch, 14795]`
- Two variants: standard ONNX (exact) and `no_dft` (DFT→MatMul, ~2e-4 tolerance)

With ONNX: Perch drops from ~80 min to ~7 min. Combined Perch+SED = ~58 min.

### Competition context

- Competition #1: 0.939 LB
- pantanal-distill V18: 0.930 LB (same architecture, scaled up)
- Our Perch ProtoSSM v4: 0.918 LB
- Our SED v5j-4fold: 0.913 LB
- 2 months until deadline

## Method

### Approach

Four phases, each with clear gates:

**Phase A**: Recover 0.918 baseline + bisect v4.2 regression
**Phase B**: Swap TF→ONNX Perch in kernel
**Phase C**: Apply pantanal-distill improvements (incremental, LB-validated)
**Phase D**: Perch + SED ensemble

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026: 59 labeled soundscapes (708 windows), ~600 test files |
| Compute | Kaggle CPU kernel (90 min budget), PACE for validation |
| Dependencies | onnxruntime, PyTorch, sklearn (no TensorFlow after Phase B) |
| Code | `user/murilo/experiments/020-perch-onnx/` |
| Branch | `mg/020-perch-onnx` |

### Procedure

#### Phase A: Recover 0.918 + Bisect Regression (1-2 days, 5 LB subs)

Current kernel in git is v4.2 (0.899). v4 (0.918) exists only as Kaggle notebook
version 1.

1. Reconstruct v4 by reverting 4 changes in v4.2:
   - Taxonomy: restore genus-based (161 groups) vs class_name (5 groups)
   - Site mapping: 0-based (v4) vs 1-based (v4.2)
   - MLP: remove n_iter_no_change=15, random_state=42
   - Ensemble: OOF-optimized weight vs fixed 0.5
2. Submit reconstructed v4 → must score 0.918
3. Bisect: 4 kernels, each applying ONE v4.2 change to v4 baseline
4. Identify culprit(s), keep safe changes, revert harmful ones

#### Phase B: Swap TF→ONNX Perch (1 day, 1 LB sub)

1. Validate ONNX locally on PACE (numerical equivalence)
2. Upload Perch ONNX model to Kaggle as dataset
3. Replace TF inference with onnxruntime in kernel
4. Submit → must score 0.918 (identical to TF)

#### Phase C: Pantanal-Distill Improvements (3-5 days, 6-7 LB subs)

Each step applies ONE change, LB-validated. Revert if regression.

| Step | Change | LB target |
|------|--------|-----------|
| C.1 | OOF-optimized ensemble weight | ≥0.918 |
| C.2 | 5-fold OOF (from 3-fold) | ≥0.919 |
| C.3 | Per-class threshold optimization | ≥0.920 |
| C.4 | Scaled probes: PCA-128, MLP(256,128) | ≥0.921 |
| C.5 | Scale ProtoSSM d=256, 3 layers | ≥0.923 |
| C.6 | Training: γ=2.5, 80ep, lr=8e-4, cosine restart | ≥0.925 |
| C.7 | Full V18: d=320, 4 layers, ResSSM d=128 | ≥0.928 |

#### Phase D: Perch + SED Ensemble (2-3 days, 2 LB subs)

After Phase C achieves ≥0.920 Perch score.

1. Create unified kernel: Perch ONNX + SED v5j-4fold ONNX
2. Runtime budget: ~7 min (Perch) + ~48 min (SED) + ~3 min (training/PP) = ~58 min
3. Ensemble: `w * perch_probs + (1-w) * sed_probs`

### Variables

- **Independent**: Perch inference backend (TF vs ONNX), ProtoSSM scale, ensemble method
- **Dependent**: LB score, kernel runtime
- **Controlled**: Labeled data (59 files, 708 windows), Perch v2 model (frozen)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| LB score | Competition metric (row-wise AUROC) |
| Kernel runtime | Must be < 90 min |
| OOF AUC | Offline quality check (but poor LB predictor) |

### Baselines

| Model | LB | Notes |
|-------|-----|-------|
| Perch ProtoSSM v4 | 0.918 | Current best Perch |
| Perch ProtoSSM v4.2 | 0.899 | Regressed, needs fix |
| SED v5j-4fold | 0.913 | Current best SED |
| pantanal-distill V18 | 0.930 | Reference (same arch, scaled) |
| Competition #1 | 0.939 | Target to approach |

### Success Criteria

- **Phase A confirm if**: Reconstructed v4 scores 0.918, regression cause identified
- **Phase B confirm if**: ONNX kernel scores 0.918 in <15 min total
- **Phase C confirm if**: Any improvement ≥0.920 (clear gain from scaling)
- **Phase C stretch**: 0.925+ (match Yuriy)
- **Phase D confirm if**: Ensemble > max(Perch alone, SED alone)
- **Phase D stretch**: 0.930+ (competitive with pantanal-distill)
- **Reject if**: ONNX produces different scores than TF (numerical divergence)

## Limitations

- Perch ONNX is community-converted — may have edge cases not covered in benchmarks
- 5 LB submissions/day constrains bisection speed
- Scaling ProtoSSM on 59 labeled files risks overfitting (pantanal-distill uses same data)
- Ensemble weight optimization on LB is noisy (±0.002 per submission)

## Leads To

- Improved Perch baseline for pseudo-label extraction (future exp)
- SED v5m-4fold ensemble variant (EfficientNet diversity)

## References

- [Perch ONNX model](https://huggingface.co/justinchuby/Perch-onnx)
- [pantanal-distill V18 annotated](../../notes/pantanal-distill-0930-annotated.md)
- [Yuriy ProtoSSM v5 annotated](../../notes/yuriy-protossm-v5-annotated.md)
- [Exp 018 Perch Probes](../018-perch-probes/tasks.md)
- [Exp 019 SED Ablations](../019-sed-ablations/tasks.md)
- [EfficientNet vs HGNetV2](../../docs/vendor/efficientnet-vs-hgnetv2.md)
