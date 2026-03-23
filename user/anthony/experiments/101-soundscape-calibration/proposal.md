# Proposal: Soundscape Label Calibration

**Status**: proposed
**Created**: 2026-03-22
**Author**: Anthony

## Hypothesis

Murilo's fine-tuned models (ProtoCLR 0.736 LB, ConvNeXt 0.720 LB, EAT 0.701 LB)
achieve per-species AUROC >0.70 on the ~47 evaluable species in labeled soundscapes,
with identifiable per-species and per-taxa accuracy profiles that determine which
species can be reliably pseudo-labeled.

## Background

Pre-computed predictions from 3 fine-tuned models exist on all 66 labeled soundscape
files (see `docs/available-predictions.md`). BirdCLEF 2026 provides 739 labeled 5s
windows — the first opportunity to measure per-species teacher accuracy on in-domain
soundscapes.

The Kaggle LB (0.70–0.74) is macro-averaged and skips classes with no true positives —
we don't know which species carry the score and which drag it down. Per-species AUROC
on labeled windows answers this directly and determines the pseudo-labeling strategy
for the 176.5h unlabeled soundscapes.

### Evaluable species

28 of 75 labeled species have zero focal training data (25 insect sonotypes + 3
amphibians). Models predict near-zero for these. Metrics are on the ~47 evaluable
species unless stated otherwise. See `docs/ground-truth-summary.md`.

### Connections

- **Follows**: Exp 100 (EDA), Exp 011/murilo (LoRA baselines)
- **Feeds into**: Pseudo-labeling 176.5h unlabeled soundscapes (priority #2)
- **Related**: `docs/exp013-diagnosis.md` (FlyingSquid failure hypotheses, out of scope)

## Method

### Setup

| Component | Details |
|-----------|---------|
| Data | 739 labeled 5s windows (exp 100 `soundscape_labels.parquet`) |
| Predictions | 3 pre-computed `.npz` files (see `docs/available-predictions.md`) |
| Compute | CPU-only, all predictions pre-computed |
| Dependencies | numpy, polars, scikit-learn, matplotlib |

### Procedure

**Phase 1: Data alignment**

1. Load `.npz` files — extract `predictions`, `seg_keys`, `species`
2. Build ground truth binary matrix from `soundscape_labels.parquet`
3. Align 792 predicted windows to 739 labeled windows; exclude 53 unlabeled gaps
4. Identify ~47 evaluable species (exclude 28 with zero focal data)

**Phase 2: Raw model evaluation** (primary deliverable)

5. Per-species AUROC and AP for each model (ProtoCLR, ConvNeXt, EAT)
6. Macro + per-taxa breakdown (Aves / Amphibia / Mammalia / Reptilia)
7. Per-site breakdown (9 sites) — detect site-specific failures
8. Inter-model agreement: pairwise correlation, species where models disagree
9. Reliability diagrams: are raw probabilities already well-calibrated?
10. Exclude species with <2 positive windows from per-species AUROC (~5 species,
    reducing effective count from ~47 to ~42). Flag but report separately.
11. Partition species into tiers:
    - **Reliable** (AUROC >0.85): pseudo-label with confidence threshold
    - **Usable** (0.70–0.85): pseudo-label with caution
    - **Unreliable** (<0.70): do not pseudo-label

**Decision gate after Phase 2:**

| Result | Action |
|--------|--------|
| AUROC >0.70, well-calibrated | Skip to Phase 4 (thresholding) |
| AUROC >0.70, miscalibrated | Proceed to Phase 3 (Platt scaling) |
| AUROC 0.55–0.70 | Phase 4 with reliable-tier species only |
| AUROC <0.55 | Reject — teachers unreliable for pseudo-labeling |

**Phase 3: Calibration** (conditional — only if Phase 2 shows miscalibration)

Split: S22 as test (~480 windows), 8 remaining sites for leave-one-site-out CV (~259
windows). See `docs/split-strategy.md`.

11. Fit Platt scaling using CV on 8 train sites:
    - Per-taxa (default), per-species where ≥5 positives, global as sanity check
12. ECE + Brier score before/after calibration
13. Evaluate on held-out S22 test set

**Phase 4: Pseudo-labeling recommendation**

See `docs/pseudo-label-strategies.md` for full comparison of 5 strategies (global
threshold + weighting, top-k ranking, temperature scaling, per-taxa isotonic,
hierarchical isotonic + residual thresholds).

14. Select strategy from the candidates in `docs/pseudo-label-strategies.md`
15. Per-species decision: reliable / usable / unreliable tier assignments
16. Produce config for applying to 176.5h unlabeled soundscapes

### Variables

- **Independent**: teacher model (3 models), aggregation (single/ensemble)
- **Dependent**: per-species AUROC, AP; macro averages; reliability diagrams
- **Controlled**: ground truth (739 windows), predictions (frozen)

## Evaluation

### Metrics

On **evaluable species only** (~47).

| Metric | Description |
|--------|-------------|
| Per-species AUROC | Discrimination ability per species |
| Macro AUROC | Mean per-species AUROC |
| Per-taxa AUROC | Grouped by Aves / Amphibia / Mammalia / Reptilia |
| Macro AP (cmAP) | Class-mean average precision |
| Reliability diagram | Visual calibration check (only metric needed to decide Phase 3) |

### Success Criteria

- **Confirm if**: macro AUROC >0.70 on evaluable species — pseudo-labels feasible
- **Partial**: 0.55–0.70 — pseudo-label only reliable-tier species
- **Reject if**: <0.55 — teachers too unreliable

## Limitations

- **~47 evaluable species**: 28 with zero focal data excluded. 159/234 species not in labeled soundscapes at all.
- **Small per-species samples**: most have ~12 windows, ~5 species have <2 positive windows (excluded from AUROC). Estimates are noisy.
- **Implicit negatives**: unlisted species assumed absent; annotations may be incomplete.
- **Pantanal-specific**: labeled sites are all Pantanal. Calibration may not transfer to other acoustic environments in unlabeled data.
