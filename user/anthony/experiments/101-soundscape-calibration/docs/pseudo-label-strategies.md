# Pseudo-Label Strategies: Decision Record

## Context

The mean ensemble of 3 fine-tuned models (ProtoCLR, ConvNeXt, EAT) achieves 0.880
macro AUROC on 42 evaluable species. The models rank species correctly but assign
compressed absolute probabilities — with 8+ species per window, correct species
often get p=0.3 instead of p=0.8. At threshold 0.5, precision is 93.8% but recall
is only 36.9%.

We need to turn ensemble probabilities into training signal for 176.5h of unlabeled
soundscapes across 234 species, most without ground truth for threshold tuning.

## Strategies

### A. Global threshold + confidence weighting

Use a single low threshold (e.g., 0.3) to generate binary pseudo-labels, then
weight each sample by the ensemble probability during downstream training.

| Aspect | Details |
|--------|---------|
| Threshold | Global, e.g., 0.3 (86% precision, 59% recall) |
| Unseen species | Same threshold for all 234 species |
| Fitting needed | None — just pick a threshold |
| Downstream use | BCE loss with sample weight = ensemble probability |

**Pros**: Simplest. No fitting. Works for all species. Confidence weighting lets
the model learn more from high-confidence windows.

**Cons**: Single threshold ignores per-taxa differences (frogs have different
probability distributions than birds). May include noisy pseudo-labels for
unreliable species.

**This is what most BirdCLEF 2025 competition winners used.**

### B. Top-k per window (rank-based)

Instead of thresholding on absolute probability, take the top-k species per window
based on ranking. k can be fixed (e.g., 8, matching the median species count in
labeled windows) or adaptive (e.g., all species with p > p_max * 0.3).

| Aspect | Details |
|--------|---------|
| Threshold | None — uses ranking, not absolute values |
| Unseen species | Handled naturally by ranking |
| Fitting needed | Choose k (or adaptive ratio) |
| Downstream use | Binary labels for top-k, negative for rest |

**Pros**: Directly exploits the strong ranking (AUROC 0.88). Immune to the
compressed probability problem. No per-species calibration needed.

**Cons**: Assumes a fixed number of species per window. Unlabeled soundscapes
may have anywhere from 0 to 20+ species — a fixed k would overpredict in quiet
windows and underpredict in dense ones. Adaptive ratio helps but adds a parameter.

### C. Temperature scaling

Learn a single temperature parameter T that divides logits before sigmoid:
`p_calibrated = sigmoid(logit / T)`. Fit T on labeled windows to minimize
cross-entropy.

| Aspect | Details |
|--------|---------|
| Parameters | 1 (temperature T) |
| Unseen species | Same T for all species |
| Fitting needed | Minimize NLL on labeled windows |
| Downstream use | Calibrated probabilities, threshold at 0.5 |

**Pros**: Simple (1 parameter). Well-studied. Directly addresses compressed
probabilities by spreading the logit distribution. After calibration, 0.5
threshold should work better.

**Cons**: Assumes all species/taxa have the same miscalibration pattern. In
practice, frogs (high co-occurrence) and birds (lower co-occurrence) likely
have different calibration curves. Single T can't capture this.

### D. Per-taxa isotonic regression

Fit one isotonic regression per taxon (Aves, Amphibia, Mammalia, Reptilia).
Pool all species within a taxon to get enough data points. Maps predicted
probabilities to observed frequencies with a monotonic step function.

| Aspect | Details |
|--------|---------|
| Parameters | Non-parametric (step function per taxon) |
| Unseen species | Use their taxon's calibration curve |
| Fitting needed | Fit on labeled windows, leave-one-site-out CV |
| Downstream use | Calibrated probabilities, threshold at 0.5 or per-species |

**Pros**: Captures taxa-specific calibration patterns. Non-parametric (no
distributional assumptions). Unseen species get taxa-level calibration.
After isotonic, 0.5 threshold is meaningful.

**Cons**: Assumes species within a taxon share calibration patterns. Thin
data for Mammalia (4 species) and Reptilia (1 species). More complex than
temperature scaling.

### E. Per-taxa isotonic + per-species residual threshold

Extend D: after per-taxa isotonic calibration, fit per-species thresholds as
offsets from 0.5 for species with enough labeled windows. Species without
data default to 0.5 on the calibrated scale.

| Aspect | Details |
|--------|---------|
| Parameters | Isotonic per taxon + threshold offset per species |
| Unseen species | Taxa isotonic + default 0.5 threshold |
| Fitting needed | Two-stage: isotonic on CV, then threshold tuning |
| Downstream use | Binary pseudo-labels at calibrated per-species thresholds |

**Pros**: Most principled. Per-taxa calibration + per-species fine-tuning where
data supports it. Reasonable defaults for unseen species.

**Cons**: Most complex. Two-stage fitting with thin data. Risk of overfitting
per-species thresholds on 12 windows. Added implementation cost for marginal gain.

## Comparison

| Strategy | Complexity | Handles unseen species | Uses ranking strength | Data needed |
|----------|-----------|----------------------|----------------------|-------------|
| A. Global + weight | Trivial | Yes (same threshold) | Indirectly (via weight) | None |
| B. Top-k ranking | Low | Yes (rank-based) | Directly | Choose k |
| C. Temperature | Low | Yes (same T) | No (calibrates abs values) | Labeled windows |
| D. Per-taxa isotonic | Medium | Yes (taxa curve) | No (calibrates abs values) | Labeled windows |
| E. Hierarchical | High | Yes (taxa + default) | No (calibrates abs values) | Labeled windows |

## Recommendation

**Start with A (global threshold + confidence weighting).** It's the simplest
approach, works for all 234 species, and is battle-tested in the competition
setting. Use threshold 0.3 for pseudo-label generation, weight by ensemble
probability in downstream training.

**If A's pseudo-labels degrade downstream training** (as exp 014 soft labels did),
try D (per-taxa isotonic) to see if better calibration fixes the problem. The
diagnostic would be: are the pseudo-labels hurting because of noise (wrong labels)
or because of calibration (right ranking, wrong probabilities)?

B (top-k) is worth exploring if per-window species density varies a lot across
sites. C and E are likely overkill given the data constraints.

## Threshold Sweep Results (2026-03-23)

CV-selected thresholds evaluated on S22 test set:

| Strategy | Threshold | Precision | Recall | F1 |
|----------|-----------|-----------|--------|-----|
| Ensemble global (CV=0.12) | 0.12 | 0.711 | 0.806 | **0.756** |
| Ensemble per-taxa | Amp=0.40, Ave=0.42 | 0.911 | 0.521 | 0.663 |
| Ensemble @ fixed 0.20 | 0.20 | 0.791 | 0.674 | 0.728 |
| Ensemble @ fixed 0.50 | 0.50 | 0.938 | 0.369 | 0.529 |

Key finding: optimal threshold is ~0.12–0.20, not 0.5. The compressed
probability problem means a low threshold with confidence weighting
(Strategy A) is the right approach.

## Decision

**Strategy A: global threshold ~0.15–0.20 + confidence weighting.**

- Threshold 0.15–0.20 for pseudo-label generation (P≈0.75–0.80, R≈0.65–0.80)
- Weight each pseudo-labeled window by ensemble probability in downstream BCE loss
- Restrict to species where fine-tuned models have signal (exclude 28 zero-focal)
- Fall back to per-taxa threshold if downstream training shows taxa-specific issues

Strategy D (per-taxa isotonic) is the next option if A's pseudo-labels degrade
downstream. Strategy B (top-k) is worth exploring if species density varies
significantly across unlabeled sites.
