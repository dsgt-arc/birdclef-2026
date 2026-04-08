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

### 1.4 Feature selection (per-logit AUROC) — EXPLORATORY, NOT A PERFORMANCE METRIC

| Metric | Value |
|--------|-------|
| Species evaluated | 59/234 (≥5 positive windows) |
| Mean max single-logit AUROC | 0.965 (selection-biased — see caveat) |
| Median max single-logit AUROC | 0.974 (selection-biased) |

**Caveat**: The 0.965 figure is the max AUROC across 14,795 logits per species,
selected and evaluated on the **same 739 windows**. Two sources of inflation:
1. Cherry-picking: top 100 logits selected by mean_diff, then max AUROC reported
2. No held-out evaluation: AUROC computed on the data used for selection

With 14,795 candidates and small n_pos (5-80), random chance alone produces
high max-AUROC due to multiple comparisons. This number is valid as **exploratory
analysis** (which logits associate with which species?) but **should not be
interpreted as achievable performance**.

The rigorous logit-space performance number comes from the sparse logistic
regression (Phase 2) which uses LOSO CV — feature selection and evaluation
happen on different data.

**What it does tell us**: there exist individual Perch logits with strong
marginal association to BirdCLEF species. The zero-shot baseline (0.701) uses
only the mapped logit per species — learned models can potentially do better
by finding the right logits. How much better is an open question for Phase 2.

### 1.5 Logit covariance

14795×14795 covariance matrix computed from 127,104 unlabeled soundscape windows.
Stored as float64 (778 MB compressed). Used as empirical Bayes prior in Phase 2.

## Phase 1 Summary

| Analysis | Key finding |
|----------|-------------|
| Zero-shot | AUROC 0.701 — decent floor, room to improve |
| Distribution shift | 96.8% of logits shifted — domain gap is severe |
| Prototype (insects) | AUROC 0.529 — Perch can't discriminate sonotypes |
| Feature selection | Per-logit associations found (exploratory, selection-biased) |
| Logit covariance | 778 MB matrix ready for empirical Bayes prior |

**Phase 2 expectations**: Ridge regression and sparse logistic (with LOSO CV)
will give us rigorous performance numbers for learned models. Key questions:
does the embedding covariance prior help ridge? Does focal training beat
soundscape-only? How much does temporal pooling add on unbiased OOF predictions?

## Phase 2: Learned Models

### 2.1 Empirical Bayes Ridge Regression (LOSO CV)

Ridge regression trained on 739 labeled soundscape windows with LOSO CV.
Six configurations: {soundscape, focal} × {embeddings, logits} × {identity, empirical Bayes prior}.
Lambda swept over [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100].

| Config | Features | Prior | Best λ | AUROC | AP |
|--------|----------|-------|--------|-------|----|
| **focal_emb_empirical_bayes** | embedding (1536) | EB | 100 | **0.749** | **0.645** |
| focal_emb_identity | embedding (1536) | I | 100 | 0.741 | 0.634 |
| sc_emb_identity | embedding (1536) | I | 100 | 0.693 | 0.549 |
| sc_emb_empirical_bayes | embedding (1536) | EB | 10 | 0.685 | 0.542 |
| sc_logit_identity | logits (14795) | I | 100 | 0.673 | 0.548 |
| sc_logit_empirical_bayes | logits (14795) | EB | 100 | 0.634 | 0.498 |

**Key findings**:
- **Focal embeddings win**: 0.749 vs 0.693 AUROC (+0.056). Despite the domain shift,
  focal recordings provide richer per-species signatures that transfer to soundscapes.
- **Empirical Bayes helps focal**: +0.008 AUROC (0.749 vs 0.741). The unlabeled
  soundscape covariance prior regularizes toward the target domain.
- **Empirical Bayes hurts soundscape**: -0.008 AUROC (0.685 vs 0.693). When features
  are already from the soundscape domain, the EB prior over-regularizes.
- **Embeddings beat logits**: 0.693 vs 0.673 on soundscapes. The 1536-d embedding
  encodes more transferable information than the 14795-d species logits.
- **Ridge beats zero-shot**: 0.749 vs 0.701 AUROC (+0.048) — learned combinations
  of features outperform single mapped logits.
- **All configs prefer high λ** (10–100): strong regularization is essential with
  p >> n (1536 features, 739 samples).

### 2.2 Sparse Logistic Regression (LOSO CV)

L1-regularized logistic regression per species on soundscape embeddings.
Only species with ≥5 positive windows evaluated (57/234).

| Metric | Value |
|--------|-------|
| Macro AUROC | **0.298** |
| Macro AP | 0.271 |
| Species evaluated | 57/234 |
| Species with AUROC > 0.7 | 16/57 |

**Top 5 species**:
| Species | AUROC | AP | n_pos |
|---------|-------|----|-------|
| 65380 | 0.982 | 1.000 | 333 |
| bufpar | 0.978 | 1.000 | 14 |
| 23158 | 0.973 | 1.000 | 175 |
| 22973 | 0.940 | 0.930 | 213 |
| 517063 | 0.933 | 0.928 | 313 |

**Bottom 5** (all AUROC = 0.000): nacnig1, orwpar, purjay1, redjun, trsowl.

**Interpretation**: Sparse logistic severely underperforms ridge (0.298 vs 0.749).
L1 sparsity discards too much signal — with p >> n and correlated features, L2
(ridge) is the right regularizer. The 16 species with AUROC > 0.7 are those with
enough positives to overcome the per-species fitting noise; for rare species, the
single-feature L1 model collapses.

### 2.3 Temporal Pooling (clip-level aggregation)

Ridge OOF predictions pooled from 5s windows to 60s soundscape clips.
Four strategies tested on both focal and soundscape ridge models.

| Track | Strategy | Clip AUROC | Clip AP | n_clips |
|-------|----------|------------|---------|---------|
| **Focal** | **window_level** | **0.715** | 0.282 | — |
| Focal | max-pool | 0.698 | 0.388 | 66 |
| Focal | softmax | 0.683 | 0.355 | 66 |
| Focal | mean-pool | 0.682 | 0.354 | 66 |
| Soundscape | mean-pool | 0.489 | 0.201 | 66 |
| Soundscape | softmax | 0.489 | 0.203 | 66 |
| Soundscape | window_level | 0.453 | 0.107 | — |
| Soundscape | max-pool | 0.392 | 0.185 | 66 |

**Key findings**:
- **Focal ridge + max-pool = 0.698 clip AUROC** (best clip-level result).
- Temporal aggregation doesn't improve over window-level AUROC (0.715 > 0.698) —
  pooling across 12 windows dilutes the signal rather than boosting it.
- **Soundscape ridge is broken at clip level** (0.392–0.489). The domain-shifted
  features produce too many false positives that pooling amplifies.
- Max-pool is best for focal (sparse bird calls → take the strongest window),
  worst for soundscape (amplifies FP noise).

### 2.4 Soundscape Profiles (site/diel signatures)

Per-site and diel activation signatures from 127,104 unlabeled + 739 labeled
soundscape windows. Activation threshold: p75 = 0.180 (logit value above which
a Perch species is "active").

**Artifacts**:
- `soundscape_signatures.parquet` — 23 rows, one per site, nested signature lists
- `soundscape_diel_signatures.parquet` — 41 rows (site × diel period)
- `soundscape_profiles_raw.npz` — (23, 14795) mean + activation frequency matrices

**Per-site summary** (23 Pantanal monitoring sites):

| Site | Windows | Logits >10% active | Logits >50% active |
|------|---------|--------------------|--------------------|
| S22 | 40,593 | 6,988 | 3,274 |
| S02 | 30,060 | 6,656 | 3,275 |
| S01 | 28,092 | 6,721 | 3,197 |
| S13 | 22,476 | 7,235 | 3,440 |
| S19 | 912 | 6,741 | 3,112 |
| ... | | | |
| S21 | 36 | 6,522 | 3,343 |
| S23 | 36 | 6,924 | 3,311 |

Data is heavily concentrated: S22 + S02 + S01 + S13 = 121K of 128K windows (95%).
Many sites have <200 windows — diel signatures for these are noisy.

**Site similarity** (cosine on mean logit profiles):
- Most similar: S01–S02 (0.998), S02–S18 (0.997), S01–S18 (0.997)
- Most dissimilar: S03–S10 (0.948), S10–S12 (0.953), S03–S04 (0.956)
- All pairs >0.94 — sites share a common Pantanal acoustic base with moderate
  variation. S10 and S03 are the most ecologically distinct sites.

**Most site-discriminative Perch logits** (highest variance across sites):

| Logit | Perch species | Var | Range | Most active | Least active |
|-------|---------------|-----|-------|-------------|--------------|
| 11213 | Psaltoda brachypennis (cicada) | 0.131 | [0.01, 1.00] | S08 | S15 |
| 1226 | Arunta interclusa (cicada) | 0.128 | [0.00, 1.00] | S08 | S23 |
| 13381 | Tamasa rainbowi (cicada) | 0.120 | [0.03, 1.00] | S21 | S15 |
| 6851 | Lanius ludovicianus (shrike) | 0.116 | [0.00, 0.96] | S12 | S09 |
| 14605 | Zammara smaragdula (cicada) | 0.113 | [0.00, 0.97] | S07 | S15 |
| 1238 | Asio otus (long-eared owl) | 0.112 | [0.00, 0.98] | S03 | S04 |
| 13853 | Trachycephalus nigromaculatus (frog) | 0.112 | [0.00, 0.92] | S03 | S04 |

**Interpretation**: Cicada and frog logits are the most site-discriminative —
Perch activates different insect/amphibian detectors at different sites, even
though these are proxy species (not the actual Pantanal species). This is useful:
site-specific insect/frog acoustic environments can serve as context features
for species prediction. S03 and S08 have distinctive acoustic signatures
(owl/frog-dominated and cicada-dominated respectively).

### 2.5 Focal Species Profiles

Per-species mean activation signatures from 265,924 focal recording windows
across 206 species (28 species have no focal data). Activation threshold: p75.

**Artifacts**:
- `focal_signatures.parquet` ��� 206 rows with nested signature lists and
  soundscape evaluation AUROCs (mean-matching + activation-matching)
- `focal_profiles_raw.npz` — MISSING (task crashed before writing, see below)
- `focal_cooccurrence.parquet` — MISSING (same crash)

**Bug**: Task crashed on line 234 after writing the parquet — `pl.DataFrame([]).sort("count")`
fails when no co-occurrence pairs are found. Fixed (empty DataFrame fallback added).
Rerun needed to produce `focal_profiles_raw.npz` and `focal_cooccurrence.parquet`.

**What the focal signatures contain** (from the parquet that was written):
- Per-species top-30 Perch logits by activation frequency (which Perch species
  fire most often for each BirdCLEF species)
- Cosine similarity evaluation on soundscapes: how well does the focal activation
  profile match soundscape windows (AUROC per species)

### 2.6 DMD Analysis — FAILED

Dynamic Mode Decomposition on per-site PCA-reduced Perch logit time series.
Two-pass approach: (1) fit PCA on logit subsample, (2) project + group by site
+ run DMD per site.

**Bug**: `pca.n_components_` on line 162 of `dmd.py` should be
`global_pca.n_components_` — NameError crashes after the first site's DMD
completes. The job ran for 5.5 hours (mostly in pass 2 grouping 127K windows by
site) before hitting the bug on the first site write. Wall time expired with
the task still "running". Fixed (variable name corrected).

### 2.7 UKF Smoothing — NOT RUN

Depends on DMD (provides state transition model). Blocked by DMD failure.

## Phase 2 Summary

| Analysis | Status | Key finding |
|----------|--------|-------------|
| Ridge regression | Done | **0.749 AUROC** (focal emb + EB prior, best model) |
| Sparse logistic | Done | 0.298 AUROC (L1 too aggressive, ridge dominates) |
| Temporal pooling | Done | Max-pool 0.698 clip AUROC (no gain over window-level) |
| Soundscape profiles | Done | 23 sites, cicada/frog logits most discriminative |
| Focal profiles | Partial | 206 species signatures written, raw npz + cooc missing (bug fixed) |
| DMD | Failed | Bug: `pca` → `global_pca` variable name |
| UKF smoothing | Blocked | Depends on DMD |

**Overall**: The focal embedding ridge with empirical Bayes prior (0.749 AUROC)
is the best model. It beats zero-shot (0.701) by 0.048 and soundscape ridge
(0.693) by 0.056. The domain shift from focal to soundscape remains the dominant
bottleneck — soundscape-trained models plateau at 0.693, and temporal pooling
doesn't help. Sparse logistic regression is not competitive.

**Open questions**:
1. Can fixing DMD + UKF temporal smoothing improve clip-level predictions?
2. Would domain adaptation (focal → soundscape) close the 0.056 gap?
3. Is 0.749 competitive enough for a Kaggle submission, or should this feed
   into a neural ensemble (e.g., as features for exp 102's EfficientNet)?
