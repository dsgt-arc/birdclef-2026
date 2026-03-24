# SoftAUC Loss

## What

Differentiable surrogate for AUROC. Since the competition metric is
macro-averaged ROC-AUC, optimizing AUC directly instead of BCE could
improve the metric we care about.

Used by BirdCLEF 2025 1st place as a key ingredient.

## How it works

AUROC = P(score(positive) > score(negative)) over all pos/neg pairs.

Non-differentiable: count how many pairs are correctly ranked (step function).
Differentiable: replace step with sigmoid.

```
SoftAUC = mean over all (pos_i, neg_j) pairs:
    sigmoid(score(pos_i) - score(neg_j))

Loss = -log(SoftAUC)  # or 1 - SoftAUC
```

Per-species, independently. Then macro-average across species.

## Computational cost

Full pairwise: O(n_pos × n_neg) per species per batch. With batch 32 and
206 species, most species have 0-1 positives per batch — so the number of
pairs is usually small. But rare species may have 0 positives in a batch,
making the loss undefined.

## Implementation options

- **Full pairwise**: exact, small batches are fine. Need to handle species
  with 0 positives (skip or use focal BCE fallback).
- **Sampled pairs**: randomly sample K pairs per species. Faster, noisy.
- **LibAUC**: library with momentum-based mini-batch AUC optimization.
  Handles the stale-gradient problem with running averages.
- **Hybrid**: focal BCE + SoftAUC weighted sum. Use BCE for gradient
  stability, SoftAUC to nudge rankings.

## Open questions

- What alpha did the 1st place use for SoftAUC vs other loss components?
- Does it work with attention pooling (which outputs probabilities, not logits)?
- How to handle species with 0 positives in a batch?
- Does it need larger batch sizes to get enough pos/neg pairs?

## Priority

Optional Phase 1 variant. Test after the main architecture ablation (Runs A/B/C)
shows which pooling mode wins. Could replace or supplement focal BCE in any phase.
