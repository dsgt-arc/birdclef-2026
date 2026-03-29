---
experiment: "018-perch-probes"
status: in-progress
date: 2026-03-27
updated: 2026-03-30
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

### Progress

- v1 (0.893): probes work but no OOF
- v2-fix (0.911): OOF + bug fixes, beat chiranjithdharma (0.910)
- v4 (pending): full ProtoSSM pipeline ported from Yuriy, targeting 0.925

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

### Phase C (done): OOF probes → 0.911 LB

1. 5-fold GroupKFold with fold-safe priors from ALL soundscape annotations
2. 6-category fuse_scores (proxy λ=0.8, prior-only replacement, -8.0 inactive)
3. K_HOUR=8, 59 fully-labeled files, sc_clean priors

### Phase D (done, awaiting LB): Full ProtoSSM v4 → target ~0.925

4. 77-dim features (PCA 64 + std/diff/interaction) + MLPClassifier(128)
5. ProtoSSMv2: 2-layer BiSSM + 4-head cross-attn + prototypical head + gated fusion
6. Training: focal BCE + distillation + mixup + SWA, 3-fold OOF
7. ResidualSSM second-pass correction (d=64, zero-init, w=0.3)
8. TTA (circular shifts [0, ±1]), OOF-optimized ensemble weight
9. 6-stage post-processing (temperature, confidence, rank-aware, delta-shift, thresholds)

Implemented as modular `src/birdclef/perch/` package (8 modules, 140 tests).
Assembled into self-contained kernel `submission_perch_protossm.py` (2756 lines).

## Runtime Budget

| Component | Est. Time | Notes |
|-----------|-----------|-------|
| Phase 1: Data + MLP probes | ~15s | OOF priors + 52 MLP probes |
| Phase 2: ProtoSSM training | ~35s | OOF + final model + ResidualSSM |
| Phase 3: Perch v2 on test | ~80 min | ~600 files, batched TF (16/batch) |
| Phase 4: Fusion + post-proc | ~1s | Ensemble + TTA + 6-stage PP |
| **Total** | **~82 min** | **Fits in 90-min budget** |

## Results

| Submission | LB | Delta |
|------------|-----|-------|
| Perch raw (no probes) | 0.836 | — |
| Perch + probes v1 (no OOF) | 0.893 | +0.057 |
| Perch + probes v2-fix (OOF) | **0.911** | +0.018 |
| ProtoSSM v4 (full pipeline) | **pending** | target +0.014 |

## Success Criteria

- ~~**Phase C**: Match chiranjithdharma (0.910)~~ **DONE** (0.911)
- **Phase D**: Approach yuriy (0.925) — **submitted, awaiting LB**

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
