---
experiment: "018-perch-probes"
status: in-progress
date: 2026-03-27
updated: 2026-03-29
author: Murilo Gustineli
---

# Experiment 018: Perch Probes

## Hypothesis

Progressively improving the Perch v2 probe pipeline — from basic LogReg (0.893)
through OOF cross-validation, richer features, MLP probes, and eventually ProtoSSM
temporal modeling — can reproduce the public leaderboard progression from 0.910 to
0.925, all within a single 90-minute Kaggle kernel using Perch-only inference.

## Motivation

### Every top notebook is Perch-only

The entire public leaderboard from 0.910 to 0.925 uses the same pattern: frozen
Perch v2 as a feature extractor, with progressively more sophisticated classifiers
on top. No public notebook uses a separate SED/EfficientNet model for these scores.

| Score | Notebook | Key technique |
|-------|----------|---------------|
| 0.910 | chiranjithdharma | LogReg probes + OOF + fold-safe priors |
| 0.913 | zirach | + ProtoSSM bidirectional SSM |
| 0.917 | pantanal-distill | + ResidualSSM + knowledge distillation |
| 0.921 | maryna | + focal loss, mixup, cross-attn, SWA, TTA |
| 0.925 | yuriy/yaroslav | + wider SSM (d=256), 6-stage post-processing |

### Why not SED+Perch ensemble?

Our SED (v4b, 0.859 LB) + Perch (~80 min) exceeds the 90-minute Kaggle kernel
budget. The Perch-only path reaches 0.925 without SED, and our SED at 0.859 would
add minimal signal on top of a 0.920+ Perch pipeline. The Perch-only approach is
simpler, faster to iterate, and has a proven progression path.

### Our current gap

Our perch-probes-v1 (0.893) matches chiranjithdharma on every hyperparameter but
lacks OOF cross-validation for probes. This causes probes to overfit the 66 labeled
files and alpha blending to be too aggressive. Closing this gap is the first step.

## Perch v2 Model

- **Kaggle source**: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`
- **Local path**: `/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu/`
- **Format**: TensorFlow SavedModel (362MB)
- **Output**: 1536-dim embeddings + 14,795-class logits per 5s window
- **Labels**: `assets/labels.csv` (14,795 scientific names + 1 non-bird class)

## Species Mapping

| Category | Count | Method |
|----------|-------|--------|
| Direct match | 203 | Scientific name match in Perch labels |
| Genus proxy | 6 | max(logits[same_genus_species]) |
| Unmapped | 25 | All insect sonotypes — score = -10.0 (suppressed) |

## Approach — Progressive Improvement

### Phase C (current): OOF probes + richer features → ~0.910

1. 5-fold GroupKFold by file with fold-safe priors
2. Upgrade features from 39-dim → 77-dim (PCA 64 + context + interactions)
3. MLP probes (sklearn MLPClassifier) replacing LogReg

### Phase D: ProtoSSM temporal modeling → ~0.917

4. Bidirectional SSM across 12 windows (d_model=128, 2 layers)
5. Gated fusion with Perch logits + prototypical classification head
6. ResidualSSM second-pass boosting

### Phase E: Post-processing pipeline → ~0.925

7. TTA (temporal shifts through ProtoSSM, no extra Perch cost)
8. Per-taxon temperature, file-level scaling, rank-aware scaling
9. Delta shift smoothing, per-class threshold sharpening

## Runtime Budget

| Component | Est. Time | Notes |
|-----------|-----------|-------|
| Load labeled embeddings | ~1s | Pre-extracted from PACE |
| Train probes/ProtoSSM | ~10-30s | On 66 labeled files |
| Perch v2 on test files | ~80 min | ~600 files, ThreadPoolExecutor |
| Apply probes + post-proc | ~5s | numpy/sklearn predict |
| **Total** | **~82 min** | **Fits in 90-min budget** |

## Baseline

- Perch raw (no probes): 0.836 LB
- Perch + probes v1 (logit, no OOF): **0.893 LB**

## Success Criteria

- **Phase C**: Match chiranjithdharma (0.910)
- **Phase D**: Match pantanal-distill (0.917)
- **Phase E**: Approach yuriy (0.925)

## Dependencies

- Exp 010 submission pipeline (Kaggle upload infrastructure)
- Perch v2 SavedModel (shared/models/perch-v2-cpu)
- 66 labeled soundscapes (for training probes)
- Pre-extracted labeled embeddings (shared/datasets/birdclef-2026/perch-labeled-embeddings)

## References

- [Perch Probes 0.910](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md)
- [Perch ProtoSSM 0.913](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill 0.917](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Maryna Optimized 0.921](../../docs/references/kaggle/birdclef-2026/notebooks/birdclef2026-optimized-maryna.md)
- [Yuriy ProtoSSM v5 0.925](../../docs/references/kaggle/birdclef-2026/notebooks/protossm-v5-time-optimised-yuriy.md)
- [Yaroslav Ensemble 0.925](../../docs/references/kaggle/birdclef-2026/notebooks/protossm-v5-ensemble-yaroslav.md)

## Leads to

- Phase D/E within this experiment (ProtoSSM, post-processing)
- If 0.920+ achieved: pseudo-labeling with Perch probes as teacher
