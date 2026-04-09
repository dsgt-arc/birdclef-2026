# Inference EDA Deep Dive: What the Model Sees and What It Misses

**Date**: 2026-04-09
**Data source**: v5j (HGNetV2-B0) inference on all train_audio (35,549 files) and
train_soundscapes (10,658 files = 66 labeled + 10,592 unlabeled).
**Inference path**: `/storage/project/ps-clef2026_birdclef-0/shared/inference/birdclef-2026/hgnet-b0-sed-v5j/`
**Reproducible script**: `user/murilo/scripts/inference_eda.py`

---

## Executive Summary

The v5j model achieves **0.9958 macro-AUC on the 75 species present in labeled soundscapes** —
near-perfect. Yet the same model scores only **0.906 on the public LB** (234 species). The
gap is almost entirely explained by the **159 species that never appear in labeled soundscape
data**. These species are trained only on focal recordings and must generalize blindly to the
soundscape domain.

This is not a model problem — it's a **data coverage problem**.

| Metric | Value |
|--------|-------|
| Macro-AUC on 75 labeled-soundscape species | 0.9958 |
| Public LB (234 species) | 0.906 (single) / 0.929 (ensemble) |
| Species in labeled soundscapes | 75 / 234 (32%) |
| Species with zero soundscape data | 159 / 234 (68%) |
| Species with zero focal data | 28 / 234 (12%) |

---

## 1. Labeled Soundscape Performance (75 species)

Per-species AUC on the 66 labeled soundscapes (739 matched segments):

| Taxonomy | Mean AUC | N Species | Min AUC | Max AUC |
|----------|----------|-----------|---------|---------|
| Aves | 0.997 | 28 | 0.982 | 1.000 |
| Insecta | 0.998 | 25 | 0.971 | 1.000 |
| Amphibia | 0.990 | 17 | 0.896 | 1.000 |
| Mammalia | 1.000 | 4 | 0.999 | 1.000 |
| Reptilia | 0.987 | 1 | 0.987 | 0.987 |
| **Overall** | **0.9958** | **75** | **0.896** | **1.000** |

**Worst species**: species 67252 (Amphibia, AUC=0.896, only 2 positive segments).
Even the "worst" species has high AUC — the model is essentially saturated on these 75 species.

**Key insight**: The 28 soundscape-only species (25 Insecta sonotypes + 3 Amphibia) all
appear in the 75 and score well. The soundscape-only species are NOT the bottleneck — they
have labeled training data in the soundscapes. The bottleneck is the 159 species that have
focal data but zero soundscape representation.

---

## 2. The 159 Unseen Species Problem

159 species have focal recordings but **never appear in any labeled soundscape**. The model
has never seen these species in the noisy, multi-species, low-SNR soundscape context that
the test set consists of. This is the core domain shift problem.

### 2.1 How the model handles unseen species on unlabeled soundscapes

| Metric | 75 Seen Species | 159 Unseen Species |
|--------|-----------------|-------------------|
| Mean max-prob per segment | 0.579 | 0.036 |
| Median max-prob per segment | 0.632 | 0.009 |
| Mean logit | -8.326 | -9.492 |
| Mean logit std | 2.298 | 1.548 |

The model is overwhelmingly biased toward the 75 seen species. For unseen species, predictions
are pushed far negative (mean logit -9.5 → sigmoid ≈ 0.00007). The logit std is also lower
for unseen species — the model is confidently saying "no" for species it hasn't seen in
soundscapes.

### 2.2 Pseudo-label coverage for unseen species

At a >0.5 probability threshold on unlabeled soundscapes:

| Coverage | Count | % of 159 |
|----------|-------|----------|
| Covered (≥1 segment) | 80 | 50% |
| Well-covered (≥10 segments) | 36 | 23% |
| Zero coverage | 79 | 50% |

**Half of the unseen species get zero pseudo-labels even at the 0.5 threshold.** These 79
species (61 Aves, 13 Amphibia, 3 Mammalia, 2 Insecta) will remain invisible to the model
unless we lower the threshold substantially or use external data.

### 2.3 Top pseudo-label candidates (unseen species)

Most-predicted unseen species on unlabeled soundscapes (>0.5 prob):

| Species | Taxon | Segments >0.5 |
|---------|-------|---------------|
| 244024 | Insecta | 221 |
| 24285 | Amphibia | 204 |
| fepowl | Aves | 198 |
| bbwduc | Aves | 186 |
| 22983 | Amphibia | 137 |
| gycwor1 | Aves | 126 |
| bncfly | Aves | 96 |
| rufnig1 | Aves | 77 |

These species have enough confident predictions to generate useful pseudo-labels. The remaining
79 species with zero coverage are the hard cases.

---

## 3. Focal Recording Analysis (train_audio)

265,924 segments from 35,549 focal recordings, 206 species.

### 3.1 Classification accuracy

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 0.716 |
| Top-5 accuracy | 0.826 |
| Mean prob for primary species | 0.627 |
| Segments with primary prob > 0.9 | 48.1% |
| Segments with primary prob < 0.1 | 25.0% |

25% of focal segments have <0.1 probability for their primary species. This means the model
is confidently wrong about the primary species in a quarter of focal recordings. This is
expected: focal recordings are 60-second files cropped into 5s chunks — many chunks contain
background noise or non-target species rather than the labeled primary species.

### 3.2 Accuracy by taxonomy

| Taxonomy | Top-1 | Top-5 | Mean Prob | N Segments |
|----------|-------|-------|-----------|------------|
| Insecta | 0.893 | 0.929 | 0.765 | 1,260 |
| Amphibia | 0.737 | 0.827 | 0.577 | 2,707 |
| Aves | 0.715 | 0.826 | 0.628 | 261,150 |
| Mammalia | 0.498 | 0.676 | 0.324 | 805 |
| Reptilia | 1.000 | 1.000 | 0.205 | 2 |

**Mammalia is the worst-performing taxon** (top-1 only 0.498, mean prob 0.324). This is likely
due to (a) very few training examples (805 segments across 8 species) and (b) mammal
vocalizations may not be well-captured by mel spectrograms optimized for birds.

**Insecta surprisingly performs best** (top-1 0.893) — the 3 insect species with focal data
have distinctive spectral signatures (continuous stridulation) that are easy to classify.

### 3.3 Worst species on their OWN focal recordings

These species have the lowest confidence when the model is shown their own recordings:

| Species | Taxon | Mean Prob | Top-1 Acc | N Segments |
|---------|-------|-----------|-----------|------------|
| 209233 | Mammalia | 0.000 | 0.000 | 5 |
| 516975 | Mammalia | 0.000 | 0.000 | 9 |
| 74580 | Mammalia | 0.001 | 0.000 | 14 |
| 23724 | Amphibia | 0.003 | 0.000 | 2 |
| 555123 | Amphibia | 0.004 | 0.000 | 24 |
| 70711 | Amphibia | 0.004 | 0.000 | 10 |
| 476521 | Amphibia | 0.041 | 0.308 | 13 |
| 24321 | Amphibia | 0.113 | 0.167 | 6 |
| 23154 | Amphibia | 0.132 | 0.408 | 49 |
| 74113 | Mammalia | 0.136 | 0.245 | 330 |

Three mammals (209233, 516975, 74580) have **zero** correct predictions. The model literally
cannot identify these species even on their own focal recordings. These are priority candidates
for data quality inspection and special handling.

### 3.4 Top confusion pairs

Most common misclassification patterns (true → predicted):

| True Species | Predicted As | Count |
|-------------|-------------|-------|
| coffal1 (Collared Forest-Falcon) | flawar1 (Flavescent Warbler) | 430 |
| compot1 (Common Potoo) | trsowl (Tropical Screech-Owl) | 243 |
| trsowl | compot1 | 233 |
| compot1 | compau (Common Pauraque) | 207 |
| compau | rufnig1 (Rufous Nightjar) | 193 |

**Nocturnal species dominate confusions**: Common Potoo ↔ Tropical Screech-Owl ↔ Common
Pauraque ↔ Rufous Nightjar. These are all nocturnal birds with similar spectro-temporal
patterns. The model struggles to distinguish them, especially in background chunks of focal
recordings.

---

## 4. Unlabeled Soundscape Landscape

10,592 unlabeled files across 21 sites, 127,104 5s segments.

### 4.1 Site distribution

| Site | Files | In Labeled? | Mean Max-Prob | Notes |
|------|-------|-------------|---------------|-------|
| S22 | 3,343 | YES | 0.783 | Dominant labeled site (40/66 files) |
| S02 | 2,505 | **NO** | 0.440 | Large new site |
| S01 | 2,341 | **NO** | 0.315 | Large new site, low confidence |
| S13 | 1,871 | YES | 0.800 | Well-represented in labeled |
| S19 | 73 | YES | 0.644 | |
| S07 | 52 | **NO** | 0.539 | |
| S06 | 54 | **NO** | 0.404 | |
| S18 | 52 | YES | 0.549 | |
| S16 | 48 | **NO** | 0.577 | |
| S10 | 46 | **NO** | 0.384 | |
| S14 | 41 | **NO** | 0.674 | |
| S15 | 39 | YES | 0.890 | High confidence — familiar site |
| ... | ... | ... | ... | |

**14 sites in unlabeled data have NO labeled counterpart.** The model has never seen labeled
soundscape data from these sites. S01 and S02 alone account for 4,846 files (46% of unlabeled)
and have low prediction confidence (0.315 and 0.440).

**Labeled sites have higher confidence**: Mean max-prob on labeled sites is ~0.7-0.9 vs
~0.3-0.5 on new sites. This confirms the site-level domain shift — the model is more
confident on familiar acoustic environments.

### 4.2 Hour distribution

Recordings are heavily skewed toward nocturnal hours (18:00-03:00), consistent with the
Pantanal soundscape monitoring schedule. Daytime recordings are sparse (06:00-17:00 combined
have only ~200 files vs ~10,000 for nighttime).

### 4.3 Site overlap: labeled ↔ unlabeled ↔ test

| Category | Sites |
|----------|-------|
| In labeled only (not in unlabeled) | S08, S23 |
| In both labeled and unlabeled | S03, S09, S13, S15, S18, S19, S22 |
| In unlabeled only (not in labeled) | S01, S02, S04, S05, S06, S07, S10, S11, S12, S14, S16, S17, S20, S21 |

Tucker Arrants (Kaggle discussion) confirmed that **train and test sites overlap**. The test
set likely includes sites from both the labeled and unlabeled sets. This means:
- Pseudo-labels from S01/S02 (the two largest unlabeled sites) would directly help with
  test-time generalization if those sites appear in the test set
- Site-level GroupKFold is "overly harsh" because the model benefits from having seen the same
  site at training time

---

## 5. Pseudo-Label Quality Estimation

### 5.1 Confidence distribution (unlabeled soundscapes)

| Threshold | Segments | % of Total |
|-----------|----------|------------|
| ≥ 0.10 | 113,852 | 89.6% |
| ≥ 0.30 | 87,787 | 69.1% |
| ≥ 0.50 | 73,040 | 57.5% |
| ≥ 0.70 | 60,374 | 47.5% |
| ≥ 0.90 | 42,453 | 33.4% |
| ≥ 0.95 | 31,823 | 25.0% |
| ≥ 0.99 | 9,805 | 7.7% |

57.5% of segments have at least one species predicted with >0.5 confidence. This is a
large pool of pseudo-label candidates.

### 5.2 Species coverage at different thresholds

| Threshold | Species Covered | Segments with ≥1 Species |
|-----------|----------------|--------------------------|
| > 0.3 | 169 / 234 (72%) | 87,787 |
| > 0.5 | 142 / 234 (61%) | 73,040 |
| > 0.7 | 122 / 234 (52%) | 60,374 |
| > 0.9 | 83 / 234 (35%) | 42,453 |

At threshold 0.5: 142 species get at least one pseudo-label. 92 species get nothing.
At threshold 0.3: 169 species covered, but lower confidence = more noise.

### 5.3 Effect of power transform (P^1.5)

| Threshold | Before | After P^1.5 | Retained |
|-----------|--------|-------------|----------|
| > 0.3 | 210,474 | 166,738 | 79.2% |
| > 0.5 | 154,697 | 127,422 | 82.4% |
| > 0.7 | 113,740 | 95,260 | 83.8% |

P^1.5 reduces predictions by ~18-21% at the 0.3 threshold, disproportionately removing
low-confidence predictions while preserving high-confidence ones. This is the intended
behavior: suppress noise, keep signal.

---

## 6. Logit Statistics (hengck23 Diagnostics)

### 6.1 Logit std by taxonomy (all soundscapes)

| Taxonomy | Mean Logit | Mean Std | N Species |
|----------|-----------|----------|-----------|
| Aves | -8.957 | 1.547 | 162 |
| Amphibia | -9.130 | 2.241 | 32 |
| Insecta | -10.800 | 1.969 | 3 |
| Mammalia | -11.054 | 2.118 | 8 |
| Reptilia | -9.930 | 2.814 | 1 |
| soundscape_only | -9.272 | 2.520 | 28 |

**Soundscape-only species have the highest logit std (2.520)**: the model's predictions for
these species vary wildly across segments. This is because the model has minimal training
data for them and the signal-to-noise ratio in soundscapes is low.

**Aves has the lowest std (1.547)**: most bird predictions are consistently pushed negative
across soundscapes, reflecting the model's confidence that most segments don't contain most
bird species (correct behavior for 162 species across 127K segments).

### 6.2 Seen vs unseen species

| Group | Mean Logit | Mean Logit Std |
|-------|-----------|----------------|
| 75 seen | -8.326 | 2.298 |
| 159 unseen | -9.492 | 1.548 |

**Seen species have higher variance (2.298 vs 1.548)**: the model actively detects these
species in some segments (high logits) and rejects them in others (low logits). Unseen species
have lower variance — the model has learned nothing soundscape-specific about them and
defaults to "no" with little variation.

### 6.3 Highest logit std species

Top 5 most variable species (highest logit std across soundscapes):

| Species | Std | Mean | Taxon | Status |
|---------|-----|------|-------|--------|
| 23158 (Dwarf Tree Frog) | 4.091 | -5.546 | Amphibia | SEEN |
| 65380 (Southern Orange-legged Leaf Frog) | 3.443 | -3.348 | Amphibia | SEEN |
| 22973 (Chaco Tree Frog) | 3.256 | -3.891 | Amphibia | SEEN |
| 555146 (Mato Grosso Snouted Tree Frog) | 3.205 | -4.402 | Amphibia | SEEN |
| 22967 (Lesser Snouted Tree Frog) | 3.023 | -5.294 | Amphibia | SEEN |

All are **Amphibia** — the frog species that dominate nocturnal soundscapes. Their high variance
reflects the model correctly detecting them in nighttime segments and rejecting them in
daytime segments. This is healthy behavior.

### 6.4 Train audio vs soundscapes (logit distribution shift)

| Domain | Mean Logit | Mean Std |
|--------|-----------|----------|
| train_audio (focal) | -9.202 | 1.681 |
| train_soundscapes (all) | -9.225 | 1.785 |

The distributions are surprisingly similar in aggregate. The domain shift is species-specific,
not global — the model's overall logit distribution hasn't shifted, but which species get
high logits changes dramatically between focal and soundscape recordings.

---

## 7. 4-Fold vs Single-Fold Pseudo-Label Coverage

Counter-intuitively, **4-fold averaging reduces pseudo-label coverage** compared to single-fold.

### 7.1 Coverage comparison (unseen species, unlabeled soundscapes)

| Threshold | Single-Fold | 4-Fold Avg | Difference |
|-----------|-------------|------------|------------|
| > 0.3 | 104 / 159 | 83 / 159 | **-21** |
| > 0.5 | 80 / 159 | 68 / 159 | **-12** |
| > 0.7 | 65 / 159 | 53 / 159 | **-12** |
| > 0.9 | 37 / 159 | 35 / 159 | **-2** |

**Why this happens**: Averaging logits across 4 folds smooths out confident individual-fold
predictions. If fold 2 detects a rare species at p=0.7 but folds 0/1/3 say p=0.01, the
average is ~0.18 — below the 0.5 threshold. The rare species detection is lost to consensus.

### 7.2 Newly covered species (4-fold gains)

Only 4 species gained coverage (0 → >0) from 4-fold averaging:

| Species | Single | 4-Fold |
|---------|--------|--------|
| 24287 | 0 | 14 |
| gretho2 | 0 | 3 |
| brcmar1 | 0 | 1 |
| eulfly1 | 0 | 1 |

### 7.3 Zero coverage in both: 75 species remain invisible

75 unseen species get zero pseudo-labels at the 0.5 threshold from BOTH single-fold and
4-fold models. These species need a fundamentally different teacher (Perch+SED ensemble)
or a much lower threshold.

### 7.4 Implication for pseudo-labeling strategy

**Use single-fold (or max-across-folds) for pseudo-labeling, not fold-averaged predictions.**
The goal of pseudo-labels is to maximize recall for rare species, not to minimize false
positives. A single confident fold detection is more valuable than a consensus "no."
Alternatively, use max-across-folds: `max(fold0, fold1, fold2, fold3)` per species per
segment to maximize coverage while still requiring at least one fold to be confident.

---

## 8. Secondary Labels & Confusion Analysis

Many top-1 "errors" on focal recordings may actually be correct detections of background
species that happen to be louder than the primary species in that 5s chunk.

### 8.1 How many "confusions" are actually secondary labels?

| Metric | Value |
|--------|-------|
| Total wrong top-1 predictions | 75,553 |
| Prediction matches a secondary label | 9,498 (12.6%) |
| Top-5 misses where ≥1 top-5 pred is secondary | 6,640 / 46,302 (14.3%) |

**12.6% of "errors" are correct background species detections.** The model is detecting
real acoustic events that happen to not be the primary label. This is actually desirable
behavior for soundscape prediction where multiple species co-occur.

### 8.2 Notable confusion pairs

| True → Predicted | Count | Is Secondary (%) | Interpretation |
|-----------------|-------|-------------------|----------------|
| coffal1 → flawar1 | 430 | 0.0% | **True confusion** — model confuses these species |
| compot1 → compau | 207 | 28.5% | **Partly correct** — Common Pauraque is background in 28% of Potoo recordings |
| pluibi1 → baymac | 104 | **100.0%** | **Fully correct** — Bay Macaw is always a secondary label when model "confuses" it with Plumbeous Ibis |
| roahaw → grekis | 107 | 10.3% | **Partly correct** — Great Kiskadee is a common background bird |

**pluibi1 → baymac is not a confusion at all** — the model correctly detects Bay Macaw in
the background of every Plumbeous Ibis recording where this "error" occurs.

### 8.3 Files with secondary labels have higher "confusion" rates

| File Type | Wrong Top-1 Rate | N Segments |
|-----------|-----------------|------------|
| With secondary labels | 0.334 | 53,389 |
| Without secondary labels | 0.271 | 212,535 |

Files with secondary labels have 23% more "errors" — but many of those errors are correct
background species detections. **The model is multi-label aware even though it's evaluated
as single-label on focal data.**

### 8.4 Implications

1. The **coffal1 ↔ flawar1 confusion** (430 cases, 0% secondary) is a true systematic error
   worth investigating — are these species acoustically similar, or is there a data quality issue?
2. The **nocturnal confusion cluster** (potoo/screech-owl/pauraque/nightjar) is partly explained
   by co-occurrence: these species genuinely co-exist in nocturnal recordings.
3. **Top-1 focal accuracy (0.716) understates the model's true ability** — ~12.6% of "errors"
   are correct multi-species detections. The effective accuracy is closer to ~0.75.

---

## 9. Focal Sample Count vs Pseudo-Label Coverage

Does having more focal training data help a species get detected in soundscapes?

### 9.1 Correlation

| Metric Pair | Spearman r | p-value |
|-------------|-----------|---------|
| Focal count vs pseudo-labels (>0.5) | **0.459** | <0.0001 |
| Focal count vs pseudo-labels (>0.3) | **0.499** | <0.0001 |
| Focal count vs max probability | **0.550** | <0.0001 |

**Moderate positive correlation** — more focal data helps, but it's not the whole story.
Many species with hundreds of focal recordings still get zero soundscape predictions.

### 9.2 Coverage by focal sample count

| Focal Count | N Species | Covered@0.5 | Covered@0.3 | Mean Max-Prob |
|-------------|-----------|-------------|-------------|---------------|
| 0-10 | 18 | 1 (6%) | 2 (11%) | 0.092 |
| 10-50 | 15 | 7 (47%) | 9 (60%) | 0.428 |
| 50-100 | 23 | 7 (30%) | 12 (52%) | 0.417 |
| 100-200 | 48 | 24 (50%) | 33 (69%) | 0.510 |
| 200-500 | 55 | 41 (75%) | 48 (87%) | 0.727 |

Species with <10 focal recordings are nearly invisible in soundscapes (6% coverage).
Species with 200+ focal recordings are well-covered (75%). But 25% of the 200+ group
STILL gets zero pseudo-labels — these are the domain shift failures.

### 9.3 Coverage by taxonomy (unseen species only)

| Taxon | N | Covered@0.5 | Mean Focal | Mean Max-Prob |
|-------|---|-------------|-----------|---------------|
| Aves | 134 | 73 (54%) | 214 | 0.561 |
| Amphibia | 18 | 5 (28%) | 8 | 0.295 |
| Insecta | 3 | 1 (33%) | 66 | 0.372 |
| Mammalia | 4 | 1 (25%) | 8 | 0.146 |

Non-bird taxa have dramatically lower coverage despite the same model. **Mammalia is
essentially invisible** (max prob 0.146 = the model never confidently predicts any mammal
in soundscapes). This confirms the mel spectrogram is optimized for birds, not mammals.

### 9.4 Data quality suspects: high focal count, zero soundscape predictions

54 species have ≥50 focal recordings but zero pseudo-labels at >0.5. Top examples:

| Species | Taxon | Focal Count | Max Prob |
|---------|-------|-------------|----------|
| houspa (House Sparrow) | Aves | 496 | 0.349 |
| socfly1 (Social Flycatcher) | Aves | 494 | 0.440 |
| squcuc1 (Squirrel Cuckoo) | Aves | 432 | 0.211 |
| swtman1 (Swallow-tailed Manakin) | Aves | 355 | 0.351 |
| baffal1 (Buff-necked Falcon) | Aves | 337 | 0.423 |
| yebela1 (Yellow-bellied Elaenia) | Aves | 297 | 0.081 |
| whbwar2 (White-bellied Warbler) | Aves | 243 | 0.034 |

These species have plenty of focal data but the model refuses to predict them in soundscapes.
Possible explanations:
- These species may genuinely not vocalize in the Pantanal soundscapes (recording sites
  don't overlap with their habitat)
- Their focal recordings may have very different acoustic conditions than soundscapes
- The model may have learned focal-specific artifacts rather than the species' true
  acoustic signature

**yebela1 and whbwar2 are the most extreme**: 297 and 243 focal recordings respectively,
but max probability of only 0.081 and 0.034 in any soundscape segment. These are strong
candidates for manual inspection.

### 9.5 Bright spots: low focal count but good soundscape generalization

| Species | Taxon | Focal Count | Pseudo-Labels@0.5 |
|---------|-------|-------------|-------------------|
| 24285 | Amphibia | 19 | 204 |
| 22983 | Amphibia | 10 | 137 |
| 22956 | Amphibia | 32 | 60 |
| 555145 | Amphibia | 9 | 36 |

All are **Amphibia** — frog species with distinctive spectral signatures that transfer well
from focal to soundscape despite minimal training data. Their continuous stridulation patterns
are spectrally unique and not easily confused with other species.

---

## 10. Implications for Data-Centric Pivot

### 10.1 The fundamental bottleneck is data coverage, not model quality

The model is near-perfect (0.9958 AUC) on the 75 species it has soundscape signal for.
The LB gap comes from the 159 species trained only on focal recordings. No amount of
training recipe tuning will fix this — the model needs **soundscape-domain data** for
these 159 species.

### 10.2 Pseudo-labeling priorities

| Priority | Species Group | Count | Strategy |
|----------|--------------|-------|----------|
| 1 | 80 unseen species with >0 pseudo-labels at 0.5 | 80 | Direct pseudo-labeling from v5j predictions |
| 2 | 79 unseen species with zero pseudo-labels | 79 | Need ensemble teacher (Perch+SED) or lower threshold |
| 3 | 75 seen species | 75 | Already well-served; pseudo-labels add marginal value |
| 4 | 28 soundscape-only species | 28 | Already in labeled soundscapes; pseudo-labels expand coverage |

**The 79 zero-coverage species are the hardest problem.** These are likely rare Aves species
that vocalize rarely in soundscapes, or species whose vocalizations overlap spectrally with
more common species. For these, we may need:
- The Perch+SED ensemble teacher (0.929) instead of v5j alone (0.906)
- Lower threshold (0.3 instead of 0.5), accepting more noise
- External data from Xeno-canto for species diversity

### 10.3 Data quality priorities

The 20 worst focal species (mean prob < 0.35) include:
- 3 mammals with literally zero correct predictions (209233, 516975, 74580)
- 6 amphibians with < 50 segments each
- Nocturnal bird confusion clusters (potoo/screech-owl/pauraque/nightjar)

These species would benefit most from:
- **VAD cleaning**: removing human speech that may confuse low-sample species
- **Manual inspection**: checking if labels are correct for the 0-accuracy mammals
- **Hard negative mining**: the nocturnal confusion cluster suggests the model needs
  explicit negative examples to distinguish similar-sounding species

### 10.4 Site diversity is a major lever

The 14 new sites in unlabeled data represent acoustic environments the model has never
seen in labeled form. S01 (2,341 files) and S02 (2,505 files) alone are massive — together
they're 6× larger than all labeled data. Pseudo-labeling these gives the model exposure
to diverse acoustic environments, which is exactly what robustness requires.

### 10.5 Use max-across-folds, not averaged, for pseudo-labeling

4-fold averaging loses 12 unseen species at the 0.5 threshold vs single-fold. For
pseudo-labeling, use `max(fold0, fold1, fold2, fold3)` per species per segment to maximize
coverage. A single confident fold detection is a better teacher signal than a smoothed-out
consensus that loses rare species.

### 10.6 ~12.6% of focal "confusions" are correct background detections

The model is already partially multi-label aware from focal recordings. This means the
model's predictions on unlabeled soundscapes may be *more* accurate than our focal-based
evaluation suggests. When we pseudo-label soundscapes, the "extra" species the model
predicts beyond the pseudo-label threshold may be real.

### 10.7 54 high-focal-count species with zero soundscape predictions need investigation

These species have plenty of training data but refuse to generalize. Before pseudo-labeling,
we should inspect: are these species genuinely absent from soundscapes, or is the model
failing to detect them? Manual listening to a few soundscapes from their known habitats
would resolve this.

### 10.8 The Perch+SED ensemble should be the teacher

The v5j model alone covers only 80/159 unseen species at threshold 0.5. The ensemble
(Perch ONNX + SED v5j-4fold) scored 0.929 on LB — Perch's pre-trained bioacoustic
knowledge likely covers species that SED misses. The ensemble should produce higher-coverage,
higher-quality pseudo-labels than v5j alone.

---

## References

| Data | Path |
|------|------|
| v5j soundscape logits | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/train_soundscapes/logits.parquet` |
| v5j focal logits | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/train_audio/logits.parquet` |
| v5j-4fold soundscape logits | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/` |
| Labeled soundscape labels | `shared/data/birdclef-2026/train_soundscapes_labels.csv` |
| Focal metadata | `shared/data/birdclef-2026/train.csv` |
| Original EDA | `docs/references/birdclef-2026-eda-analysis.md` |
| EDA notebook | `user/murilo/notebooks/003-birdclef2026-eda.ipynb` |
| Analysis script | `user/murilo/scripts/inference_eda.py` |
