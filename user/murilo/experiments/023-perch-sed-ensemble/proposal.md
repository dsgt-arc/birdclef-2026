# Experiment 023: Perch + SED Ensemble

## Hypothesis

Combining Perch ProtoSSM v18 (0.925 LB) with SED v5j-4fold-pp2 (0.917 LB) via
score-level fusion will exceed either individual model's LB score, potentially
reaching 0.930+. The models are complementary: Perch generalizes across acoustic
domains (frozen embeddings, never fine-tuned on focal data) while SED is strong
on species with focal recordings (trained from scratch on BirdCLEF data).

## Background

### Current State
- **Perch ProtoSSM v18**: 0.925 LB (best overall, two-kernel TF pipeline)
- **SED v5j-4fold-pp2**: 0.917 LB (best SED, single ONNX kernel)
- **Competition #1**: 0.939 LB (target)

### Why Ensembling Should Work
Across 40+ experiments (exp 006-022), **only input representation changes and
inference-time combinations have improved LB**. Training recipe changes are 0-for-14.
Ensembling is an inference-time combination — the one category that consistently works.

The two models are fundamentally different:
- **Perch**: Frozen Perch v2 embeddings → MLP probes + BiSSM temporal model → 6-stage PP
- **SED**: HGNetV2-B0 backbone trained from scratch → LSEHead → TTA + priors

They process audio differently (Perch: raw waveform → pretrained encoder; SED: mel
spectrogram → CNN) and make different types of errors (Perch: domain-general but
lower species discrimination; SED: strong focal species but weaker domain transfer).

### Timing Constraint
Both pipelines must run in a single Kaggle kernel under 90 minutes (CPU-only).
Perch TF takes ~65 min alone, so we use **Perch ONNX** (~8 min, 9x faster).
This incurs a ~0.005 LB penalty (pretrained artifacts trained on TF embeddings),
but the ensemble gain should more than compensate.

## Method

### Approach
Score-level fusion: run each pipeline with its own full post-processing, then
take a weighted average of the final probability vectors.

### Setup

| Parameter | Value |
|-----------|-------|
| Data | BirdCLEF 2026 test soundscapes (~600 files) |
| Compute | Kaggle CPU kernel (90 min limit, ~16GB RAM) |
| Perch model | Perch ONNX (391MB) + pretrained ProtoSSM V18 artifacts |
| SED model | HGNetV2-B0 v5j × 4 folds ONNX (13MB each) + pp2 priors |
| Ensemble | Weighted average: W_PERCH * perch + (1-W) * sed |
| Branch | `mg/023-ensemble` |

### Procedure

**Phase A: Simple weighted average (w=0.5)**
1. Build single Kaggle kernel combining Perch ONNX + SED pipelines
2. Each pipeline runs its own full post-processing independently
3. Weighted average with w=0.5 (equal weight)
4. Submit to LB

**Phase B: Weight grid search**
5. Submit w=0.3 and w=0.7
6. Pick optimal weight based on LB scores

**Phase C: Advanced ensembling (if needed)**
7. Per-taxa weighting (different w for Aves vs Amphibia vs Insecta)
8. Calibration adjustments
9. PP order experiments

### Variables

**Independent**: Ensemble weight W_PERCH ∈ {0.3, 0.5, 0.7}
**Dependent**: Kaggle LB score (row-wise AUC)
**Controlled**: Both model pipelines frozen (no retraining), same test files,
same post-processing within each pipeline

## Evaluation

### Metrics
- Primary: Kaggle LB score (macro-averaged per-row AUC)
- Secondary: Runtime (must be < 90 min)

### Baseline

| Model | LB |
|-------|-----|
| Perch ProtoSSM v18 (TF) | 0.925 |
| Perch ProtoSSM v18 (ONNX) | 0.920 |
| SED v5j-4fold-pp2 | 0.917 |

### Success Criteria
- **Success**: Ensemble LB > 0.925 (beats best individual model)
- **Strong success**: Ensemble LB > 0.930 (significant ensemble gain)
- **Rejection**: Ensemble LB < 0.920 (ensemble hurts vs max individual)

## Limitations

1. **ONNX penalty**: Using Perch ONNX (~0.920) instead of TF (~0.925) due to
   timing constraints. The ~0.005 penalty comes from pretrained artifacts being
   trained on TF embeddings.
2. **No OOF validation**: Cannot tune ensemble weight offline because Perch and
   SED have different fold structures (5-fold vs 4-fold). Must tune on LB.
3. **Simple fusion**: Score-level averaging is the simplest approach. More
   sophisticated methods (stacking, feature concatenation) would require
   retraining infrastructure.

## References

- [Exp 020: Perch ONNX](../020-perch-onnx/) — Perch ProtoSSM pipeline, ONNX validation
- [Exp 021: SED Training](../021-sed-training/) — SED training experiments, pattern analysis
- [Exp 019: SED Ablations](../019-sed-ablations/) — v5j recipe, 4-fold inference
- [Pattern Analysis](../../notes/2026-04-06-experiment-pattern-analysis.md) — evidence for inference-time improvements
