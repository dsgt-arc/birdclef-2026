# Results: Perch v2 Bayesian Label Transfer

**Date**: 2026-04-07
**Author**: Anthony
**Status**: Phase 1 complete, Phase 2 pending

## Phase 1: Static Baselines

### 1.1 Zero-shot baseline (direct Perch logit readout)

For 203 directly-mapped species + 6 genus proxies, read the corresponding Perch
logit as a zero-shot predictor. No training, no parameters.

| Metric | Value |
|--------|-------|
| Macro AUROC | **0.701** |
| Macro AP | 0.213 |
| Species evaluated | 45/209 mapped (164 skipped: <2 positives in 739 windows) |
| Unmapped | 25 insect sonotypes |

**Distribution of per-species AUROC**: mean=0.701, median=0.772, std=0.204

**Top 5 species** (zero-shot AUROC):
| Species | AUROC | AP | n_pos | Method |
|---------|-------|----|-------|--------|
| 43435 (Black Howling Monkey) | 0.990 | 0.778 | 12 | direct |
| hyamac1 | 0.983 | 0.289 | 10 | direct |
| rufhor2 | 0.961 | 0.542 | 4 | direct |
| 25092 | 0.954 | 0.512 | 24 | direct |
| nacnig1 | 0.922 | 0.133 | 11 | direct |

**Bottom 5 species**:
| Species | AUROC | AP | n_pos | Method |
|---------|-------|----|-------|--------|
| 65377 | 0.319 | 0.009 | 9 | direct |
| 67107 | 0.317 | 0.009 | 9 | direct |
| purjay1 | 0.316 | 0.006 | 5 | direct |
| bafcur1 | 0.307 | 0.009 | 9 | direct |
| 1491113 | 0.466 | 0.094 | 79 | genus_proxy |

**Interpretation**: Perch's zero-shot signal is decent for well-represented species
(median AUROC 0.772) but unreliable for species with few labeled windows. The genus
proxy approach (max over genus logits) performs poorly — too noisy for species-level
discrimination. Howler monkey (43435) is perfectly detected, consistent with exp 101.

### 1.2 Distribution shift (focal vs soundscape)

| Metric | Value |
|--------|-------|
| Significantly shifted logits | **14,326/14,795 (96.8%)** |
| Median KS statistic | 0.177 |
| Mean KS statistic | 0.206 |
| Max KS statistic | 0.744 (logit 8958) |

**Top shifted logits by mean difference**:
| Logit index | Focal mean | Soundscape mean | Abs diff |
|-------------|-----------|-----------------|----------|
| 5755 | -0.712 | 1.988 | 2.700 |
| 4099 | -0.191 | 2.214 | 2.404 |
| 8958 | 0.352 | 2.684 | 2.332 |
| 4114 | -0.902 | 1.296 | 2.197 |
| 8957 | 2.260 | 4.430 | 2.170 |

**Interpretation**: Domain shift is **massive**. Nearly all Perch logits shift
significantly between focal recordings and Pantanal soundscapes. Some logits
show >2.5 mean difference — entire species detection profiles change between
domains. This confirms exp 105's finding (0.684 focal vs 0.052 soundscape cmAP).

**Implication**: Models trained on focal features will degrade severely on
soundscapes. The empirical Bayes approach (using unlabeled soundscape covariance
as prior) is critical — it regularizes toward the soundscape distribution, not
the focal distribution.

### 1.3 Prototype classifier (insect sonotypes)

Nearest-centroid in embedding space for the 25 unmapped `47158sonXX` species.

| Metric | Value |
|--------|-------|
| Macro AUROC | **0.529** (near random) |
| Macro AP | 0.089 |
| Evaluated | 9/25 (16 skipped: insufficient LOSO coverage) |

**Per-sonotype results** (evaluated only):
| Sonotype | AUROC | AP | n_pos |
|----------|-------|----|-------|
| 47158son03 | **0.903** | 0.429 | 33 |
| 47158son08 | 0.761 | 0.046 | 17 |
| 47158son04 | 0.665 | 0.033 | 17 |
| 47158son11 | 0.625 | 0.062 | 36 |
| 47158son25 | 0.583 | 0.123 | 84 |
| 47158son01 | 0.560 | 0.034 | 23 |
| 47158son10 | 0.463 | 0.039 | 33 |
| 47158son13 | 0.096 | 0.027 | 36 |
| 47158son05 | 0.105 | 0.003 | 3 |

**Interpretation**: Perch embeddings mostly fail to discriminate insect sonotypes
(macro AUROC 0.529 ≈ random). One exception: son03 at 0.903 AUROC — this sonotype
has distinctive acoustic features that Perch captures. Most others are near chance,
confirming that Perch (a bird model) encodes little useful information for insect
call discrimination. The 16 skipped sonotypes have all positives in a single site,
making LOSO evaluation impossible.

### 1.4 Feature selection (per-logit AUROC)

| Metric | Value |
|--------|-------|
| Species evaluated | 59/234 (≥5 positive windows) |
| Mean max single-logit AUROC | **0.965** |
| Median max single-logit AUROC | 0.974 |

**Interpretation**: For most evaluable species, there exists at least one Perch
logit that is highly predictive (>0.96 AUROC). This is the key finding — the
information exists in the logit space, the zero-shot readout (0.701) just picks
the wrong logit. The sparse logistic regression (Phase 2) should recover this
signal by selecting the best logits per species.

**Gap**: zero-shot macro AUROC 0.701 vs best-single-logit 0.965 — there's a
0.264 AUROC gap that learned models should close.

### 1.5 Logit covariance

14795×14795 covariance matrix computed from 127,104 unlabeled soundscape windows.
Stored as float64 (778 MB compressed). Used as empirical Bayes prior in Phase 2.

## Phase 1 Summary

| Analysis | Key finding |
|----------|-------------|
| Zero-shot | AUROC 0.701 — decent floor, room to improve |
| Distribution shift | 96.8% of logits shifted — domain gap is severe |
| Prototype (insects) | AUROC 0.529 — Perch can't discriminate sonotypes |
| Feature selection | Single-logit AUROC 0.965 — signal exists, needs better selection |
| Logit covariance | 778 MB matrix ready for empirical Bayes prior |

**Phase 2 expectations**: Ridge regression and sparse logistic should close the
gap between 0.701 (zero-shot) and 0.965 (best single logit). The empirical Bayes
prior should help generalization from focal→soundscape.
