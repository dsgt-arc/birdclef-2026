# Experiment 021: SED Training — Results

## Phase A: Site/Hour Priors

### v5j-pp2: 0.909 LB (+0.003)

Priors added +0.003 over the v5j baseline (0.906 → 0.909). Below the +0.006
expected from Salman's report.

| Submission | LB | Runtime | Notes |
|-----------|-----|---------|-------|
| v5j (baseline, exp 019) | **0.906** | ~20 min | No priors |
| **v5j-pp2** | **0.909** | 11 min | + site/hour priors |
| v5j-4fold (baseline, exp 019) | **0.913** | ~50 min | No priors |
| v5j-4fold-pp2 | pending | — | Submitting next |

### Why only +0.003 (not +0.006)?

Possible explanations for the smaller-than-expected prior boost:

1. **Different prior implementation**: Our exp 010 applies priors multiplicatively
   in probability space (`prob *= prior^lambda`). Salman may use logit-space
   blending (like exp 020's `fuse_scores_with_tables`) which is more nuanced
   with 6-category taxa handling and different lambda per category.

2. **Lambda tuning**: Our lambdas are fixed (Aves=0.3, Amphibia/Insecta=0.5,
   Mammalia/Reptilia=0.2). Salman may have tuned these on a validation set.

3. **Model strength matters**: v5j (0.906) may already be capturing some
   site/time patterns that priors are redundant with. Salman's base model
   (0.917 → 0.923) may have different failure modes that priors correct better.

4. **Test set sites**: If the hidden test set includes sites not in our 9
   training sites, priors fall back to global rates (minimal effect).

### Potential follow-up

- Try exp 020's logit-space `fuse_scores_with_tables` approach on SED output
- Tune lambda values via cross-validation
- But +0.003 suggests priors are a minor factor — training improvements (Phase B)
  are likely more impactful
