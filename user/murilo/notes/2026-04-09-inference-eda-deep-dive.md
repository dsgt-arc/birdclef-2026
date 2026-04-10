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

**Important caveat on the 0.9958**: This metric is evaluated on the same 66 labeled
soundscape files the model trained on — there is no held-out soundscape validation set.
The model may be partly overfitting to the 9 training sites' acoustic conditions
(especially Site 22, which has 40/66 files). Evidence:
- 739 labeled segments is tiny — easy to memorize
- Site 22 dominates: the model could learn "S22 background + frog = species X" rather
  than generalizable species detection
- Gaurav Rawat's data shows site-level GroupKFold gives erratic CV — consistent with
  site-specific overfitting

However, the 0.9958 is not purely overfitting:
- Perch (frozen, never trained on our soundscapes) also scores 0.918-0.925 LB on these species
- 47/75 species also have focal training data providing independent signal
- Train/test sites overlap (Tucker Arrants), so site-specific learning partly helps on LB

**The real problem remains the 159 unseen species** — even discounting the 0.9958 as partly
inflated, those species have literally zero soundscape signal. Pseudo-labeling from diverse
sites (especially the 14 new unlabeled sites) both addresses the 159-species gap AND breaks
any site-specific overfitting by exposing the model to new acoustic environments.

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

## 7. Single vs Avg-4Fold vs Max-4Fold Pseudo-Label Coverage

Three strategies for generating SED pseudo-labels from 4-fold models, with dramatically
different coverage for rare/unseen species.

### 7.1 Coverage comparison (unseen species, unlabeled soundscapes)

| Threshold | Single-Fold | Avg-4Fold | **Max-4Fold** | Max vs Single |
|-----------|-------------|-----------|---------------|---------------|
| > 0.1 | 128 / 159 | 116 / 159 | **146 / 159** | **+18** |
| > 0.3 | 104 / 159 | 83 / 159 | **125 / 159** | **+21** |
| > 0.5 | 80 / 159 | 68 / 159 | **109 / 159** | **+29** |
| > 0.7 | 65 / 159 | 53 / 159 | **80 / 159** | **+15** |
| > 0.9 | 37 / 159 | 35 / 159 | **55 / 159** | **+18** |

**Max-across-folds is the clear winner**: +29 species at >0.5 compared to single-fold,
and +41 compared to averaging. Averaging HURTS (-12 vs single) because it smooths out
confident individual-fold predictions for rare species.

**Why max-folds works**: If fold 2 detects a rare species at p=0.7 but folds 0/1/3 say
p=0.01, averaging gives ~0.18 (lost), but max gives 0.7 (kept). Each fold sees a
slightly different training set, so different folds catch different rare species.

### 7.2 Total pseudo-label predictions

| Threshold | Single | Avg-4Fold | Max-4Fold |
|-----------|--------|-----------|-----------|
| > 0.3 | 210,474 | 148,413 | **311,590** |
| > 0.5 | 154,697 | 108,143 | **223,122** |
| > 0.7 | 113,740 | 77,997 | **163,471** |

Max-folds produces ~1.5× more pseudo-label predictions than single-fold. This is not just
noise — it's rare species detections from individual folds that the other folds missed.

### 7.3 Species recovered by max-folds (single=0, max>0 at >0.5)

**34 species newly covered** that single-fold missed entirely:

| Species | Taxon | Segments (max) | Notes |
|---------|-------|----------------|-------|
| 24287 | Amphibia | 36 | |
| spbant3 | Aves | 13 | Was in 54 "data quality suspects" (205 focal) |
| gretho2 | Aves | 11 | |
| fotfly | Aves | 7 | |
| tattin1 | Aves | 6 | |
| blchaw1 | Aves | 5 | |
| scadov1 | Aves | 4 | |
| houspa | Aves | 2 | Was in 54 suspects (496 focal!) |
| swtman1 | Aves | 2 | Was in 54 suspects (355 focal) |
| ... (25 more) | ... | 1-3 | |

Notable: several of the 54 "data quality suspect" species (high focal count, zero single-fold
coverage) are recovered by max-folds — they DO exist in soundscapes, just barely.

### 7.4 Per-fold unique contributions

| Fold | Unseen Species Detected | Unique to This Fold |
|------|------------------------|-------------------|
| 0 | 88 | 6 |
| 1 | 82 | 2 |
| 2 | 78 | 5 |
| 3 | 84 | 5 |

Every fold contributes unique species that no other fold detects. This confirms that
max-across-folds is the right strategy — dropping any single fold loses species.

### 7.5 Zero coverage summary

| Strategy | Zero Coverage at >0.5 | Zero at >0.3 |
|----------|----------------------|--------------|
| Single-fold | 79 / 159 | 55 / 159 |
| Avg-4fold | 91 / 159 (worse!) | — |
| **Max-4fold** | **50 / 159** | **34 / 159** |
| All three combined | 45 / 159 | — |

**Max-folds cuts the zero-coverage count nearly in half** (79 → 50 at >0.5).
At the more permissive >0.3 threshold, only **34 species remain truly hard** — these
are the ones that need Perch or external data.

### 7.6 The 34 truly hard species (zero at >0.3, even max-folds)

By taxonomy: 18 Aves, 11 Amphibia, 3 Mammalia, 2 Insecta.

These species are invisible to the SED model under any fold or threshold strategy.
They are the primary targets for the Perch teacher — if Perch's pre-trained bioacoustic
knowledge can detect even some of these, the ensemble teacher is justified.

### 7.7 Implication for pseudo-labeling strategy

**Use max-across-folds (`max(fold0, fold1, fold2, fold3)`) as the SED teacher component.**

- Covers 109/159 unseen species at >0.5 (vs 80 for single, 68 for avg)
- Produces 223K predictions at >0.5 (vs 155K single)
- Each fold contributes unique rare species detections
- Apply power transform (P^1.5) after max to control false positive rate
- For the 50 remaining zero-coverage species, Perch is needed

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

### 10.2 Pseudo-labeling priorities (updated after Phase A)

Using **max-across-folds** as the SED teacher component:

| Priority | Species Group | Count | Strategy |
|----------|--------------|-------|----------|
| 1 | Unseen species covered by SED max-folds at >0.5 | 109 | Direct pseudo-labeling from max(fold0..fold3) |
| 2 | Unseen species with zero SED coverage but possibly covered by Perch | 50 | Need Perch teacher — Phase B/C/D will determine coverage |
| 3 | **Truly hard species** (zero even at >0.3 with max-folds) | 34 | Need Perch or external data — SED cannot detect these |
| 4 | 75 seen species | 75 | Already well-served; pseudo-labels expand site/hour coverage |
| 5 | 28 soundscape-only species | 28 | Already in labeled soundscapes; pseudo-labels add diversity |

**The 50 zero-coverage species (at >0.5 with max-folds) are the primary target for Perch.**
Phase D (Perch vs SED comparative EDA) will determine how many Perch recovers. If Perch
covers 30+ of these, the ensemble teacher is clearly justified. If only 5-10, we need
lower thresholds or external data.

**Preview opportunity**: Existing Perch mapped scores at
`shared/embeddings/birdclef-2026/perch-v2/labeled/perch_mapped_scores.parquet` could give
a quick comparison of Perch vs SED on the 75 labeled-soundscape species before Phase B
completes.

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

### 10.5 Use max-across-folds for pseudo-labeling (Phase A finding)

Max-across-folds recovers **+29 unseen species** at >0.5 (80 → 109/159), cutting zero-coverage
from 79 to 50. At >0.3, only 34 species remain truly hard. Each of the 4 folds contributes
2-6 unique species. **34 new species are recovered** that single-fold missed entirely —
including several of the 54 "data quality suspects" (House Sparrow, Swallow-tailed Manakin),
proving they DO exist in soundscapes but are too rare for any single fold to catch reliably.

For pseudo-labeling: `max(fold0, fold1, fold2, fold3)` per species per segment, then
apply power transform P^1.5 to control false positive rate.

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

Using SED max-across-folds, 109/159 unseen species are covered at >0.5. The remaining 50
need Perch. The ensemble (Perch + SED v5j-4fold) scored 0.929 on LB — Perch's pre-trained
bioacoustic knowledge should cover species that SED misses.

**Phase D (pending, blocked on Phase B) will answer the critical questions:**
1. How many of the 50 SED-zero species does Perch cover?
2. How many of the 34 truly hard species (zero at >0.3) does Perch cover?
3. What's the union: `max(SED-4fold) + Perch MLP probes` total coverage?
4. For species both detect, do they agree on which segments? (quality signal)
5. Does Perch do better on the 14 unlabeled-only sites? (site diversity)

If Perch covers 30+ of the 50 zero-coverage species, the combined teacher covers
~140+/159 — a viable foundation for pseudo-labeling. If Perch covers <10, we need
external data or much lower thresholds.

### 10.9 Existing Perch data we haven't fully exploited

We already have Perch 234-class mapped scores for the 66 labeled soundscapes at
`shared/embeddings/birdclef-2026/perch-v2/labeled/perch_mapped_scores.parquet`.
This was generated in exp 018 but never compared per-species against SED.
A quick comparison could preview Perch vs SED complementarity on the 75 labeled
species before Phase B completes.

---

## 11. Discussion Notes & Nuances

Captured from follow-up Q&A on 2026-04-09. These clarify subtleties that the raw numbers
don't convey.

### 11.1 Why mean max-prob matters for unseen species

Mean max-prob of 0.036 for 159 unseen species means: on a typical soundscape segment, the
model's MOST confident prediction for ANY of the 159 unseen species is 3.6%. The model has
learned "soundscapes contain the 75 species I know — nothing else." It isn't uncertain
about unseen species — it's **confidently wrong**.

Even the 80 species that DO get some pseudo-labels have sparse, scattered detections (the
top one has only 221 segments out of 127K). The model occasionally catches these species
but treats them as anomalies, not expected inhabitants. This means pseudo-labels for the
80 partially-covered species will be thin — the bulk of pseudo-labels will go to the 75
already-seen species where the model is already saturated.

The 79 zero-coverage species need a **fundamentally different teacher** — the v5j SED model
cannot provide pseudo-labels for them at any reasonable threshold. The Perch+SED ensemble
(0.929 LB) has access to Perch's pre-trained bioacoustic knowledge for species v5j has
never learned to detect in soundscapes. This is the strongest argument for using the
ensemble as the teacher, not v5j alone.

### 11.2 The 14 unlabeled-only sites and what they mean for pseudo-labeling

The labeled soundscape data covers 9 sites. The unlabeled data covers 21 sites, of which
14 are completely new — no labeled data exists for them. S01 (2,341 files) and S02 (2,505
files) are the two largest and have LOW model confidence (0.315 and 0.440 mean max-prob).

Pseudo-labeling these files serves two purposes:
1. **Species-in-context data**: what species X sounds like at site S01 at 2am, mixed with
   whatever background noise that specific site has
2. **Site diversity**: the model learns that the same species can sound different across
   acoustic environments (different vegetation, distance to mic, ambient noise levels)

Both directly address the domain shift that causes the focal→soundscape gap. Tucker Arrants
(Kaggle discussion) confirmed that train and test sites overlap, so the test set likely
includes some of these 14 new sites. Training on pseudo-labeled data from S01/S02 could
directly improve predictions on those sites if they appear in the test set.

### 11.3 How to ensure pseudo-label correctness (layers of defense)

There is no way to guarantee pseudo-labels are correct — that's inherent to semi-supervised
learning. The defense is layering multiple quality controls:

1. **Teacher quality**: Our 0.929 ensemble is much stronger than the 0.767 teacher that
   failed in exp 015. Stronger teacher = better pseudo-labels.
2. **Confidence thresholding**: Only keep predictions above a threshold (0.5 or 0.7).
   Higher threshold = fewer but more accurate labels.
3. **Power transform (P^1.5)**: Non-linearly suppresses uncertain predictions. P=0.6
   becomes 0.465 (killed). P=0.9 becomes 0.854 (kept). Selectively removes borderline
   predictions where the model is most likely wrong.
4. **Soft labels with loss weighting**: Use probabilities as soft targets, not hard 0/1.
   Weight pseudo-label loss lower than ground truth (e.g., w=0.3-0.5 vs w=1.0). Limits
   damage from bad pseudo-labels.
5. **Multi-iteration refinement**: Train student → student becomes teacher → better
   pseudo-labels → repeat (Nikita used 3-4 iterations).
6. **Density-weighted sampling**: Segments with too many high-confidence predictions
   (dense chorus → confused) are downweighted. Segments with a few clean, high-confidence
   predictions are upweighted.

Even imperfect pseudo-labels for 80 species are better than the current zero soundscape
representation for those species. The model literally has no soundscape signal for 68% of
species — any signal, even noisy, is an improvement.

### 11.4 Why 4-fold averaging hurts coverage but helps submission

These are two different contexts with opposite requirements:

**Submission (inference on test set)**: For the 75 well-known species (which dominate the
LB score), averaging 4 folds **reduces noise and improves calibration**. If fold 1 says
"frog at 0.85" and fold 3 says "0.75", averaging gives 0.80 — a more stable estimate.
Better calibration → better AUC ranking. That's why 4-fold (0.913) beats single (0.906).

**Pseudo-labeling (teacher predictions for training)**: Here we need to **detect rare
species** that individual folds might catch but the average misses. If fold 2 detects a
rare bird at 0.7 but the other three say 0.01, averaging gives 0.18 — the detection is
lost. For rare species, a single confident fold detection is more valuable than consensus.

Prescription:
- **Submission**: keep averaged 4-fold (better calibration)
- **Pseudo-labeling**: use `max(fold0, fold1, fold2, fold3)` per species per segment
  (maximize rare species recall)

The Kaggle submission pipeline (confirmed in `submission_ensemble.py:2058`) uses
`fold_probs.mean(axis=0)` — correct for submission. The pseudo-labeling pipeline
should use max instead.

### 11.5 Multi-label pseudo-label capture strategy

The approach: **store the full soft probability vector for every segment, filter at
training time.**

1. **Teacher inference**: Run ensemble on all 127K unlabeled segments. Save 234-dim
   probability vector per segment (sigmoid outputs).
2. **Power transform + threshold**: P^1.5, then keep species above threshold (e.g., 0.3).
   If the model predicts species A at 0.85, B at 0.72, C at 0.15, after P^1.5 these
   become 0.78, 0.60, 0.06. With threshold 0.3, the label is `A=0.78, B=0.60, others=0`.
3. **Multi-label BCE loss**: Each species is an independent binary prediction. The pseudo-
   label provides a soft target per species. The model learns "this segment contains A and
   B" — matching the multi-label reality of soundscapes.

The key question — how many species per segment to keep:
- **Threshold approach** (recommended): Keep all species above threshold after power
  transform. Flexible — some segments have 1 species, some have 8. Matches the EDA
  finding that soundscapes average 4.2 species per segment.
- **Top-N approach**: Keep only the N most confident. Risk: clips high-N segments.
- **Soft labels**: Keep all 234 probabilities. Simplest but noisiest.

The secondary label finding (12.6% of "confusions" are real) supports this: the model's
multi-label predictions on soundscapes may be more trustworthy than focal metrics suggest.
The "extra" species it detects beyond the primary are often genuinely present.

### 11.6 Data augmentation vs pseudo-labeling for domain-shift species

For the 54 species with 200+ focal recordings but zero soundscape predictions:

**Augmentation** (adding noise to focal recordings) was tested globally in v6c (exp 021)
and regressed on LB. Training recipe changes have been 0-for-14. However, *targeted*
augmentation (only for failing species) hasn't been tested — this is a more nuanced
approach that might work.

**The deeper issue**: Some of these species (e.g., House Sparrow — urban bird) may
genuinely not vocalize in Pantanal soundscapes. In that case, the correct prediction IS
zero, and augmenting their focal data would inject false positives. **Manual inspection
of the 54 suspects is needed before trying to "fix" them.**

**Pseudo-labeling is still the better first path**: It gives the model actual soundscape
examples of species (if they exist), not synthetically noisified focal recordings. The
real soundscape context (background species, site-specific noise, reverberation) is hard
to simulate with augmentation but trivially captured by pseudo-labeling.

The two approaches aren't mutually exclusive: augmentation on pseudo-labeled soundscape
data could help further in later iterations, once we've confirmed which species actually
occur in soundscapes.

---

## 12. Phase C: Perch 234-Class Predictions (2026-04-10)

Applied the full Perch→BirdCLEF pipeline from the exp 023 submission kernel to Phase B
embeddings + logits for all 127,896 soundscape segments.

### 12.1 Pipeline details

The MLP probes require **142 features per species** (not just PCA embeddings):
- 128 PCA-compressed embedding dimensions
- 14 contextual: raw Perch logit, prior logit, fused base score, 5 temporal features
  (prev/next/mean/max/std within file), 3 diffs, 3 interaction terms

Full pipeline: Perch 14,795 logits → 234 BirdCLEF mapping (203 direct, 3 genus proxy,
28 unmapped sonotypes) → prior fusion (site/hour) → temporal smoothing → 58 MLP probes
→ alpha blend (0.45) → sigmoid.

### 12.2 Embedding validation

Phase B TF embeddings are **identical** to the embeddings used to train the probes in
exp 020 (cosine similarity = 1.000000 on the 792 labeled soundscape segments). Both use
the same Perch v2 TF SavedModel. No retraining needed.

### 12.3 Coverage results

| Metric | Raw Perch (no priors) | Full Pipeline |
|--------|----------------------|---------------|
| Species with max prob >0.5 | 195 / 234 | 212 / 234 |
| Species with max prob >0.3 | 199 / 234 | 220 / 234 |
| Species with max prob >0.9 | 189 / 234 | 201 / 234 |
| True zeros (<0.001) | 28 (sonotypes) | 6 |

The 28 unmapped sonotypes have no Perch taxonomy match and rely entirely on prior tables.
The full pipeline recovers most via site/hour priors.

### 12.4 Pipeline behavior on inactive species

The pipeline **suppresses unmapped inactive species** to -8.0 (sigmoid ≈ 0.0003), which
discards valid genus proxy signal. Example: species 1595929 (Uruguay Harlequin Frog) has
raw Perch max prob 0.41 via genus proxy, but gets set to 0.0003 because it's unmapped and
not in labeled soundscapes. Use `perch_234_raw_probs.parquet` (before prior suppression)
for fair detection comparison in Phase D.

Script: `user/murilo/scripts/apply_perch_probes.py`

---

## 13. Phase D: Perch vs SED Comparative EDA (2026-04-10)

The critical analysis comparing Perch and SED predictions on the same 127K soundscape
segments. Uses Perch raw probabilities (no prior suppression) vs SED max-across-folds.

Script: `user/murilo/scripts/perch_sed_comparison.py`

### 13.1 Coverage: Perch rescues 40 SED-zero species at >0.5

| Threshold | SED Max-Folds | Perch Raw | Perch Rescues |
|-----------|---------------|-----------|---------------|
| > 0.1 | 146 / 159 | 155 / 159 | +11 |
| > 0.3 | 125 / 159 | 153 / 159 | +28 |
| > 0.5 | 109 / 159 | 149 / 159 | +40 |
| > 0.7 | 80 / 159 | 146 / 159 | +66 |
| > 0.9 | 55 / 159 | 145 / 159 | +90 |

Perch dramatically outperforms SED on unseen species coverage. At every threshold, Perch
covers more species. At >0.5, Perch rescues **40 species** that SED max-folds missed entirely.

### 13.2 Truly hard species: Perch covers 28/34

Of the 34 species invisible to SED even at >0.3 with max-folds:

| Perch Threshold | Species Covered |
|-----------------|-----------------|
| > 0.1 | 30 / 34 |
| > 0.3 | 28 / 34 |
| > 0.5 | 24 / 34 |

Only **6 species** remain invisible to both models (all Amphibia + 1 Mammalia):
23150, 23154, 23724, 25214, 476521 (Amphibia), 74580 (Mammalia).

These 6 species have <50 focal recordings each and extremely low-SNR vocalizations. They
may require external data or specialized handling.

### 13.3 Union coverage exceeds success criteria

| Threshold | SED | Perch | Union | Union (all 234) |
|-----------|-----|-------|-------|-----------------|
| > 0.1 | 146/159 | 155/159 | 157/159 | 231/234 |
| > 0.3 | 125/159 | 153/159 | **153/159** | 222/234 |
| > 0.5 | 109/159 | 149/159 | **149/159** | 216/234 |

**Success criterion: ≥120/159 unseen at >0.3 → 153/159 ✓ PASS**

Union at >0.3 doesn't improve over Perch alone because **every species SED detects, Perch
also detects**. There are zero SED-only species at any threshold. Perch is a strict superset
of SED for species coverage.

### 13.4 Complementarity: zero SED-only species

| Group (at >0.3) | Count | Interpretation |
|------------------|-------|----------------|
| Both detect | 125 | Core species — both models agree |
| SED only | **0** | SED adds nothing for coverage |
| Perch only | 28 | Perch exclusively detects these |
| Neither | 6 | Truly hard — need external data |

At >0.5: Both=109, SED-only=0, Perch-only=40, Neither=10.

### 13.5 CRITICAL: Perch calibration problem

**Perch's raw probabilities are severely miscalibrated.** The coverage numbers above are
real (Perch CAN detect signal for those species) but the precision is terrible:

| Teacher | Pseudo-labels at >0.5 (unseen species, unlabeled soundscapes) |
|---------|--------------------------------------------------------------|
| SED max-folds | 5,033 |
| Perch raw | **15,091,757** |

Perch produces **3,000× more pseudo-labels** than SED. Many "rescued" species show Perch
max=1.000 across 120K+ of 127K segments — Perch thinks the species is present in >95% of
all recordings. Example: Social Flycatcher (`socfly1`) at >0.5 in 127,017/127,104 segments.
This is a diurnal bird — it cannot be present in all nocturnal recordings.

**Root cause**: Perch was trained on 14,795 classes. When a BirdCLEF species maps to a
common taxon in Perch's training data, the raw logit is permanently positive regardless
of whether the species is actually vocalizing. The sigmoid output reflects Perch's prior
that the taxon exists somewhere, not that it's detected in this specific segment.

**Implication for teacher strategy**: Perch raw probabilities are NOT directly usable as
pseudo-labels. The teacher must either:
1. Use SED as primary teacher + Perch only for SED-zero species
2. Apply aggressive thresholding (>0.9 or >0.95) to Perch predictions
3. Use Perch for species DETECTION (binary: present/absent per file) rather than
   segment-level pseudo-labels
4. Calibrate Perch probabilities against SED or labeled soundscapes before use

### 13.6 Per-species agreement: low correlation (models are complementary)

For the 125 species both detect at >0.3:

| Metric | Value |
|--------|-------|
| Spearman ρ mean | 0.223 |
| Spearman ρ median | 0.222 |
| Spearman ρ range | -0.099 to 0.572 |

Low correlation confirms the models make **different predictions on different segments**.
This is good for ensemble diversity but means Perch and SED "see" different acoustic events.

Highest agreement: Amphibia and distinctive Aves (frogs, nightjars — unique spectral signatures).
Lowest agreement: common Aves where Perch gives flat high scores while SED is segment-specific.

### 13.7 Per-site analysis: Perch dominates everywhere

| Site Group | N Sites | SED Mean Coverage | Perch Mean Coverage |
|-----------|---------|-------------------|---------------------|
| Labeled | 7 | 16.3 / 159 | 145.3 / 159 |
| Unlabeled-only | 14 | 12.6 / 159 | 145.9 / 159 |

Perch covers ~145 unseen species per site vs SED's ~12-16. The Perch advantage is equally
strong on new (unlabeled) sites as on familiar (labeled) sites — consistent with Perch's
domain-general pretrained knowledge not depending on site-specific training.

SED shows a modest labeled-site advantage (16.3 vs 12.6) — it generalizes slightly better
to sites it has seen labeled data for. But this is dwarfed by Perch's advantage.

### 13.8 Ensemble teacher: SED for precision, Perch for recall

The D.7 analysis reveals the optimal teacher strategy is NOT a simple weighted average:

| Strategy | Coverage@0.5 | Pseudo-labels@0.5 | Issue |
|----------|-------------|-------------------|-------|
| SED only | 109/159 | 5,033 | Misses 50 species |
| Perch only | 149/159 | 15,091,757 | Massive FP rate |
| w=0.5 blend | 143/159 | 178,734 | Middle ground |
| max(SED,Perch) | 149/159 | 15,091,758 | Same as Perch alone |

**Recommended strategy**: Tiered teacher

1. **SED-covered species** (109/159 at >0.5): Use SED max-folds as teacher.
   SED has 5K precise pseudo-labels. Apply P^1.5 power transform.
2. **Perch-only species** (40 species): Use Perch predictions BUT with aggressive
   filtering — require per-file consistency (species present in ≥N/12 windows),
   high threshold (>0.9), or site-level frequency caps.
3. **Neither species** (10 at >0.5, 6 at >0.3): No teacher available. These 6-10
   species must rely on focal training only or external data.
4. **Seen species** (75): Already well-served. Pseudo-labels from both models expand
   site/hour coverage.

### 13.9 Updated implications for exp 025

Phase D **passes both success criteria**:
- ✅ Union covers 153/159 unseen species at >0.3 (target: ≥120)
- ✅ Perch covers 40 of 50 SED-zero species at >0.5 (target: ≥30)

But the calibration finding changes the pseudo-labeling design:
- Cannot use Perch raw probabilities directly (15M FPs)
- Need tiered teacher: SED-primary + Perch-rescue for SED-blind species
- Perch predictions need calibration or aggressive filtering before use
- The 6 truly invisible species (5 Amphibia + 1 Mammalia) need external data

### 13.10 Taxonomy breakdown: Perch's Aves saturation

Coverage by taxonomy class (unseen species, unlabeled soundscapes):

| Taxon | N Unseen | SED @>0.5 | Perch @>0.5 | Neither @>0.3 |
|-------|----------|-----------|-------------|---------------|
| Aves | 134 | 102 (76%) | **134 (100%)** | 0 |
| Amphibia | 18 | 6 (33%) | 10 (56%) | **5** |
| Insecta | 3 | 1 (33%) | 2 (67%) | 0 |
| Mammalia | 4 | 0 (0%) | 3 (75%) | **1** |

**Perch covers 100% of unseen Aves** — every single one. But this is misleading:
- Mean Perch probability across all Aves × segments: **0.721**
- 123/134 species have mean prob > 0.5 (predicted as "present" in the average segment)
- Species mean prob range: 0.29 to 0.97

Perch is not detecting individual Aves — it's giving all Aves species a permanently
high probability. The Perch v2 model was trained on a taxonomy dominated by bird species,
and its logits reflect a strong prior that "there are birds" rather than "this specific
bird is vocalizing now."

**Amphibia is the hardest taxon**: 5/18 unseen species invisible to both models. These
are the same species flagged in D.2 as truly hard — low-SNR frog vocalizations with
few focal recordings.

### 13.11 Calibration quantification: 3,000x false positive rate

Species predicted per 5-second segment (unseen species, unlabeled soundscapes):

| Threshold | SED Mean | SED Max | Perch Mean | Perch Max |
|-----------|----------|---------|------------|-----------|
| > 0.3 | 0.1 | 3 | **138.4** | 147 |
| > 0.5 | 0.04 | 3 | **118.7** | 137 |
| > 0.7 | 0.02 | 2 | **82.1** | 119 |
| > 0.9 | 0.01 | 2 | **30.1** | 73 |

A typical soundscape segment contains 2-5 vocalizing species. SED's predictions (mean
0.04 unseen species per segment at >0.5) are consistent with this — it rarely predicts
unseen species and when it does, it's selective. Perch predicts **119 unseen species per
segment** — obviously wrong.

Pseudo-label density per covered species at >0.5:

| Teacher | Species | Median Segments | Mean Segments | Total PLs |
|---------|---------|-----------------|---------------|-----------|
| SED | 109 | 8 | 46 | 5,033 |
| Perch | 149 | **119,976** | **101,287** | **15,091,757** |

**126/159 species** appear in >50% of all Perch segments. **82/159** appear in >90%.
This confirms Perch raw predictions are essentially constant per species — not
segment-specific detection.

Even at >0.9, Perch predicts 30 species per segment. A power transform alone cannot
fix this — the problem is that Perch logits are permanently positive for common taxa.

### 13.12 Labeled soundscape validation: SED dominates on known species

Ground truth comparison using 792 labeled soundscape segments (75 species):

| Metric | SED Max-Folds | Perch Raw |
|--------|---------------|-----------|
| Macro-AUC | **0.9987** | 0.7311 |

Precision (fraction of positive predictions that are correct):

| Threshold | SED Precision | SED Predictions | Perch Precision | Perch Predictions |
|-----------|---------------|-----------------|-----------------|-------------------|
| > 0.3 | **0.890** | 3,435 | 0.059 | 31,124 |
| > 0.5 | **0.941** | 3,194 | 0.067 | 26,062 |
| > 0.7 | **0.979** | 2,958 | 0.083 | 19,316 |
| > 0.9 | **0.997** | 2,616 | 0.139 | 9,717 |

**SED is near-perfect** on known species (94% precision at >0.5, 0.9987 AUC).
**Perch has ~6% precision** — 94% of its predictions are false positives. Even at >0.9,
Perch precision is only 14%.

Per-species AUC comparison: **SED is better on 62/70 species, Perch on 0/70.** No single
species has better Perch AUC than SED AUC. This confirms SED is a strictly better predictor
on known species.

**Implication**: On the 75 seen species, SED should be the sole teacher. Perch adds nothing
but noise. Perch's value is entirely in the **40 SED-blind species** where SED has zero
coverage — there, any signal (even noisy) is better than nothing.

---

## 14. Phase E: Ecological Sanity Check (2026-04-10)

### 14.1 All 54 original suspects are resolved

The 54 species with ≥50 focal recordings but zero single-fold SED predictions at >0.5
are ALL recovered by either SED max-folds or Perch:

| Recovery Method | Count |
|-----------------|-------|
| SED max-folds (>0.5) | 32 |
| Perch only (>0.5) | 22 |
| Still problematic | **0** |

The "data quality suspect" framing from section 9.4 is resolved — these species are not
data quality issues. They are rare-in-soundscapes species that single-fold SED missed
but multi-fold or Perch can detect. No ecological sanity check needed for the 54.

### 14.2 The 6 truly invisible species: data scarcity, not ecological absence

The 6 species invisible to both models at >0.3 all have **1-5 focal recordings**:

| Species | Taxon | Common Name | Focal | SED Max | Perch Max | In Pantanal? |
|---------|-------|-------------|-------|---------|-----------|-------------|
| 23150 | Amphibia | Central Dwarf Frog | 1 | 0.057 | 0.003 | **Yes** (wet season nocturnal) |
| 23154 | Amphibia | Bahia Dwarf Frog | 5 | 0.208 | 0.010 | **No** (NE Brazil only) |
| 23724 | Amphibia | Waxy Monkey Tree Frog | 1 | 0.006 | 0.278 | **Yes** (Chaco/Pantanal, nocturnal) |
| 25214 | Amphibia | Muller's Termite Frog | 3 | 0.262 | 0.190 | **Yes** (explosive breeder, distinctive call) |
| 476521 | Amphibia | Cuyaba Dwarf Frog | 3 | 0.163 | 0.011 | **Yes** (type locality is Cuiabá!) |
| 74580 | Mammalia | Black-tailed Marmoset | 3 | 0.002 | 0.000 | **Yes** but **diurnal** |

**Ecological assessment**:
- **4/6 are ecologically expected** in Pantanal nocturnal soundscapes (P. centralis,
  P. sauvagii, D. muelleri, P. nattereri). The Cuyaba Dwarf Frog was literally named
  after the Pantanal capital.
- **1 is out of range**: P. albifrons (Bahia Dwarf Frog) is restricted to NE Brazil.
  If it appears in test predictions, it's likely confusion with congeners P. centralis
  or P. nattereri that ARE present.
- **1 is present but diurnal**: M. melanurus (Black-tailed Marmoset) is documented in
  Pantanal but only active during daylight. Since soundscapes are mostly nocturnal, its
  absence from predictions is expected. Could appear in dawn/dusk segments only.

### 14.3 Root cause: data scarcity, not domain shift

Unlike the original 54 suspects (which had 50-496 focal recordings and suffered from
focal→soundscape domain shift), the 6 invisible species have **1-5 focal recordings**.
With this little training data, neither SED nor Perch can learn the species' acoustic
signature well enough to detect it in noisy soundscapes.

These species cannot be helped by pseudo-labeling (no teacher can detect them), better
augmentation, or training recipe changes. They need **more training data** — either
external recordings from xeno-canto/Macaulay Library, or discovery in soundscapes
through manual annotation.

### 14.4 Implications

1. The ecological sanity check confirms the data-centric thesis: the model's failures
   are **data problems** (coverage, scarcity), not model quality problems.
2. The 54 original suspects are all recoverable with existing models — no manual
   intervention needed.
3. The 6 invisible species are a hard floor: 4 are genuinely present in Pantanal but
   have too few recordings. 1 is out of range. 1 is diurnal.
4. For exp 025, these 6 species should be excluded from pseudo-label evaluation — they
   cannot improve regardless of teacher strategy.

---

## References

| Data | Path |
|------|------|
| v5j soundscape logits | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/train_soundscapes/logits.parquet` |
| v5j focal logits | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/train_audio/logits.parquet` |
| v5j-4fold soundscape logits | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/` |
| Perch TF embeddings + logits | `shared/inference/birdclef-2026/perch-v2-tf/{train_soundscapes,train_audio}/` |
| Perch 234-class raw probs | `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_raw_probs.parquet` |
| Perch 234-class full pipeline | `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet` |
| Perch labeled embeddings (exp 018) | `shared/embeddings/birdclef-2026/perch-v2/labeled/` |
| MLP probes + ProtoSSM artifacts | `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/` |
| Labeled soundscape labels | `shared/data/birdclef-2026/train_soundscapes_labels.csv` |
| Focal metadata | `shared/data/birdclef-2026/train.csv` |
| Original EDA | `docs/references/birdclef-2026-eda-analysis.md` |
| EDA notebook | `user/murilo/notebooks/003-birdclef2026-eda.ipynb` |
| Analysis script (SED) | `user/murilo/scripts/inference_eda.py` |
| Analysis script (Perch vs SED) | `user/murilo/scripts/perch_sed_comparison.py` |
| Perch probe application | `user/murilo/scripts/apply_perch_probes.py` |
| Perch inference script | `user/murilo/inference/perch_inference.py` |
| Original EDA | `docs/references/birdclef-2026-eda-analysis.md` |
| EDA notebook | `user/murilo/notebooks/003-birdclef2026-eda.ipynb` |
| Analysis script | `user/murilo/scripts/inference_eda.py` |
| Perch inference script | `user/murilo/inference/perch_inference.py` |
