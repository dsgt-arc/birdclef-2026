# Exp 013 Soft Label Failure: Diagnostic Hypotheses

Exp 014/015 used FlyingSquid soft labels as training targets → 0.237 cmAP vs 0.336
hard-label baseline. **Diagnosing this is out of scope for exp 101** (requires
BirdNET/Perch 2026 embeddings that don't exist yet). These hypotheses are documented
for a future follow-up if needed.

## Hypotheses

**A. Domain shift in probes**: Probes trained on 2025 focal embeddings may not
transfer to 2026 soundscapes.
Test: compare probe AUROC focal vs soundscape.

**B. Conditional independence violation**: Co-occurring species (median 8/window)
cause correlated LF votes, inflating triplet method accuracy estimates.
Test: compare estimated vs measured LF accuracy.

**C. Prevalence prior mismatch**: Focal prevalence (~0.5% per species) vs soundscape
(8+ species co-occurring) biases the posterior.
Test: compare per-species positive rates across domains.

**D. Threshold domain shift**: Percentile thresholds set on focal logit distributions
may discretize differently on soundscape distributions.
Test: compare LF logit distributions across domains.

**E. Inherent soft-label weakness**: Even calibrated soft labels may provide weaker
gradients than hard labels for a frozen linear probe.
Test: use well-calibrated labels from exp 101 on the same downstream task.
