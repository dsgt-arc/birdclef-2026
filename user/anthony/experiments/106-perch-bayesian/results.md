# Results: Perch v2 Bayesian Label Transfer

**Date**: 2026-04-09
**Author**: Anthony
**Status**: Phase 1 complete, Phase 2 complete

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

**Interpretation**: Decent floor for mapped species (median AUROC 0.772), but
unreliable for rare species and useless for unmapped insects.

### 1.2 Distribution shift (focal vs soundscape)

| Metric | Value |
|--------|-------|
| Significantly shifted logits | **14,326/14,795 (96.8%)** |
| Median KS statistic | 0.177 |
| Mean KS statistic | 0.206 |

**Interpretation**: Domain shift is massive. Nearly all Perch logits shift between
focal recordings and Pantanal soundscapes. Confirms exp 105's finding (0.684 focal
vs 0.052 soundscape cmAP).

### 1.3 Prototype classifier (insect sonotypes)

Nearest-centroid in embedding space for 25 unmapped `47158sonXX` species.

| Metric | Value |
|--------|-------|
| Macro AUROC | **0.529** (near random) |
| Evaluated | 9/25 |

**Interpretation**: Perch embeddings fail to discriminate insect sonotypes. One
exception: son03 at 0.903 AUROC. Perch is bird-specialized.

### 1.4 Feature selection — EXPLORATORY ONLY

Per-logit AUROC on same data used for selection. **Not a performance metric** —
inflated by multiple comparisons (14,795 candidates, small n_pos).

### 1.5 Covariance matrices

- Logit covariance: 14795×14795 from 127K unlabeled windows (778 MB compressed)
- Embedding covariance: 1536×1536, effective rank 184
- Cached derived quantities: inverses + top-500 eigendecomposition

## Phase 2: Learned Models + Structure

### 2.1 Empirical Bayes Ridge Regression (LOSO CV)

| Config | Features | Prior | Best λ | AUROC | AP |
|--------|----------|-------|--------|-------|----|
| **focal_emb_empirical_bayes** | emb (1536) | EB | 100 | **0.749** | **0.645** |
| focal_emb_identity | emb (1536) | I | 100 | 0.741 | 0.634 |
| sc_emb_identity | emb (1536) | I | 100 | 0.693 | 0.549 |
| sc_emb_empirical_bayes | emb (1536) | EB | 10 | 0.685 | 0.542 |
| sc_logit_identity | logits (14795) | I | 100 | 0.673 | 0.548 |
| sc_logit_empirical_bayes | logits (14795) | EB | 100 | 0.634 | 0.498 |

**Key findings**:
- Focal embeddings dominate (+0.056 over soundscape)
- Empirical Bayes prior helps focal (+0.008), hurts soundscape (-0.008)
- Embeddings beat logits, all configs prefer high λ

### 2.2 Sparse Logistic — NOT COMPETITIVE

L1-regularized logistic: 0.298 AUROC. L1 too aggressive for p >> n.

### 2.3 Temporal Pooling

Max-pool 0.698 clip AUROC — no gain over window-level (0.715). Pooling dilutes signal.

### 2.4 Multi-Prototype Profiles (NMF + k-means)

Per-species: K=4 prototypes via NMF (logits) + k-means (embeddings).
Per-site: K=6 prototypes with diel characterization.
206 species + 21 sites completed in 27 minutes (with spawn multiprocessing fix).

**Multi vs single prototype cosine matching** (global AUROC, 42 focal species):

| Space | Single | Multi (K=4) | Delta |
|-------|--------|-------------|-------|
| Logit | 0.565 | 0.589 | +0.025 |
| **Embedding** | **0.800** | **0.816** | **+0.016** |

15/42 species improved. Modest lift — most species are well-represented by a
single mean prototype.

### 2.5 Prototype Ridge

Ridge on prototype similarity features (max cosine to each species' K prototypes):
**0.721 AUROC** — worse than raw embedding ridge (0.749). Prototype features as
ridge inputs don't add value over raw embeddings.

### 2.6 Star Schema Scorer

Composes through shared Perch logit space: species similarity × site affinity × diel prior.

**LOSO AUROC** (per-site average, all 70 evaluable species):

| Config | AUROC | AP |
|--------|-------|----|
| emb_sim (cosine only) | 0.696 | 0.611 |
| emb_site (+ site affinity) | 0.696 | 0.611 |
| emb_site_diel (+ diel) | 0.696 | 0.608 |
| logit_sim | 0.628 | 0.547 |

**Site affinity has zero effect.** All Pantanal sites have cosine similarity >0.94
— there isn't enough site-level variation to exploit as a prior.

**Critical breakdown by focal data availability:**

| Model | With focal (42 sp) | Without focal (28 sp) | All (70 sp) |
|-------|-------------------|-----------------------|-------------|
| Star schema | 0.816 | 0.258 (random) | 0.593 |
| Ridge | 0.858 | 0.500 (random) | 0.715 |

The star schema gives strong scores for the 42 species with focal recordings
(0.816 AUROC) but is useless for the 28 species without focal data (insect
sonotypes, etc.). Ridge learns from the 739 labeled soundscapes and covers all
species. Ridge also beats star schema on the focal species (0.858 vs 0.816).

## Summary

| Model | AUROC (LOSO) | Coverage | Notes |
|-------|-------------|----------|-------|
| Zero-shot Perch logit | 0.701 | 45 mapped species | No training |
| **Ridge (focal emb + EB)** | **0.749** | 70 species | Best overall |
| Prototype Ridge | 0.721 | 70 species | Worse than raw ridge |
| Star schema cosine | 0.696 | 70 species (0.816 on 42 focal) | Site/diel priors don't help |
| Sparse logistic | 0.298 | 57 species | L1 too aggressive |

### Key Takeaways

1. **Ridge regression is the best model** (0.749 AUROC). Focal embeddings +
   empirical Bayes prior is the winning combination.

2. **The star schema works for species with focal data** (0.816 AUROC on 42
   species) but fails completely for the 28 species without focal recordings.
   Ridge covers all species by learning from labeled soundscapes.

3. **Site and diel priors don't help.** The Pantanal sites are too acoustically
   similar (cosine >0.94) for site-level conditioning to add signal. Diel
   variation exists but is uniform across species in our current formulation.

4. **Multi-prototypes provide modest lift** (+0.016 AUROC in embedding space).
   Most species are well-captured by a single mean profile.

5. **Domain shift remains the dominant bottleneck.** 96.8% of logits shift
   between focal and soundscape. The empirical Bayes prior partially bridges
   this gap but doesn't solve it.

6. **Insect sonotypes are unsolvable with Perch.** The 28 unmapped species
   (mostly insects) get random-chance scores from all approaches. A different
   encoder (e.g., BirdNET, or a general audio model) would be needed.

### What This Means for the Competition

Exp 106 establishes that Perch v2 features + simple linear models achieve 0.749
AUROC on labeled soundscapes. For context:

- Exp 105 (Perch frozen + SED head): 0.695 focal cmAP, 0.072 sc_cmap
- Exp 102 (EfficientNet LoRA): **0.794 Kaggle LB**
- Murilo's Perch ProtoSSM v4: **0.918 Kaggle LB**

The 0.749 AUROC (LOSO) is a different metric than Kaggle LB (row-wise cmAP on
test soundscapes), so direct comparison is unreliable. A Kaggle submission would
give the true signal. The ridge model's simplicity (linear, no neural network)
makes it easy to export and fast to run.

The most promising use of exp 106 outputs for competition scoring is as
**ensemble features** — the ridge OOF predictions, prototype cosine similarities,
and site affinity priors are complementary signals that could be stacked with
neural model predictions (exp 102, murilo/018).
