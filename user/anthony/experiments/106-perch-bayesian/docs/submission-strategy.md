# Submission Strategy: Bayesian Ensemble Calibration

## Overview

Each Kaggle submission is a noisy observation of model performance on the true
test distribution. We use these observations to calibrate mixing weights across
multiple scoring spaces and select the best ensemble for final submission.

## Available scoring spaces

Each space runs the full star schema pipeline independently (prototypes, site
states, domain transfer, scoring). After composing through the mediating
dimension, all produce the same shape: (S, W) scores per species per window.

| Space | Mediating dim | Status | Notes |
|-------|---------------|--------|-------|
| P (Perch logits) | 14,795 | Running | Interpretable, 203 mapped species |
| E (Perch embeddings) | 1,536 | Designed | Compact, all species, needs E-space pipeline run |
| F_P (finetuned logits) | varies | Future | From exp 102 EfficientNet or finetuned Perch |
| F_E (finetuned embeddings) | varies | Future | Same model, penultimate layer |
| Ridge (learned) | 1,536→234 | Done (0.749) | Trained on soundscape labels directly |

## Per-species Bayesian mixing

Not all spaces are equally useful for all species:

- **Well-mapped birds** (203 species with direct Perch mapping): P-space strong,
  E-space complementary
- **Genus-proxy species** (6 species): P-space noisy (genus-level signal), E-space
  may be better
- **Unmapped insects** (25 species): P-space near-zero signal, E-space only path
- **Rare species** (few focal windows): prototypes unreliable in any space,
  ridge regression may win

Mixing weights are species-dependent:

```
π_m(s) = exp(auroc_m(s) / τ) / Σ_m' exp(auroc_m'(s) / τ)
```

Estimated from per-species AUROC on the 739 labeled soundscape windows.

## Calibration from LB submissions

### Information from each submission

| Observable | Granularity | What it tells you |
|------------|-------------|-------------------|
| Local AUROC | Per-species | Model quality on labeled distribution |
| Public LB | Aggregate | Model quality on test distribution |
| Local↔LB gap | Aggregate | How well local evaluation transfers |

### Calibration protocol

**Phase 1: Diverse model submissions** (early)

Submit each space independently to establish per-space LB baselines:

1. P-space multi-prototype scoring → LB_P
2. E-space multi-prototype scoring → LB_E
3. Ridge regression → LB_ridge
4. Domain-corrected P-space → LB_P_corrected

Each submission costs one daily attempt but gives you the local↔LB gap for
that space. A space with high local AUROC but low LB is overfitting to the
labeled distribution — down-weight it in the ensemble.

**Phase 2: Calibrated ensembles** (middle)

Use the per-space LB scores to set ensemble weights:

```python
# Observed: local AUROC and LB score for each space
# The LB-to-local ratio calibrates trust
trust_m = LB_m / local_auroc_m   # >1 means transfers well, <1 means overfits

# Adjust mixing weights
calibrated_π_m(s) = π_m(s) * trust_m
calibrated_π_m(s) /= Σ_m' calibrated_π_m'(s)  # renormalize
```

Submit ensembles with calibrated weights. Compare ensemble LB to best single-space LB.

**Phase 3: Final selection** (late)

Track all submissions:

```
Submission   Local   Public_LB   Gap      Strategy
P-space      0.759   ???         ???      Multi-prototype, domain-corrected
E-space      ???     ???         ???      Multi-prototype in embedding space
Ridge        0.749   ???         ???      Empirical Bayes ridge
Ensemble-v1  ???     ???         ???      Equal weights
Ensemble-v2  ???     ???         ???      Calibrated weights
```

Select the submission with highest public LB for final. If ensemble consistently
beats individual spaces, the Bayesian averaging is working.

## Thompson sampling interpretation

This is equivalent to Thompson sampling for ensemble weights:

1. **Prior**: π_m(s) from local per-species AUROC
2. **Action**: submit an ensemble with specific weights
3. **Reward**: public LB score
4. **Update**: posterior on mixing weights given observed LB

Each submission explores a region of the weight space. The LB score updates
the posterior. Over time, the posterior concentrates on the weight configuration
that maximizes LB performance.

The key constraint is the daily submission limit — each attempt is expensive.
The optimal strategy prioritizes high-information submissions (diverse individual
models early, targeted ensemble variations later) over incremental tweaks.

## Connection to star schema

The submission strategy is itself a composition problem. The "mediating dimension"
is the per-species score vector (234 dimensions). Each space produces a score
vector, and the ensemble composes them through the species dimension:

```
final_score = Σ_m  diag(π_m) @ score_m   # (234,) weighted sum
```

The mixing weights `π_m` are the "transfer weights" between spaces, analogous
to the domain transfer weights between focal and soundscape. The LB submissions
calibrate these weights, just as the KS statistics calibrate the per-dimension
domain transfer weights.
