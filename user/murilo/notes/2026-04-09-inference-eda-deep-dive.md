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

## 7. Implications for Data-Centric Pivot

### 7.1 The fundamental bottleneck is data coverage, not model quality

The model is near-perfect (0.9958 AUC) on the 75 species it has soundscape signal for.
The LB gap comes from the 159 species trained only on focal recordings. No amount of
training recipe tuning will fix this — the model needs **soundscape-domain data** for
these 159 species.

### 7.2 Pseudo-labeling priorities

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

### 7.3 Data quality priorities

The 20 worst focal species (mean prob < 0.35) include:
- 3 mammals with literally zero correct predictions (209233, 516975, 74580)
- 6 amphibians with < 50 segments each
- Nocturnal bird confusion clusters (potoo/screech-owl/pauraque/nightjar)

These species would benefit most from:
- **VAD cleaning**: removing human speech that may confuse low-sample species
- **Manual inspection**: checking if labels are correct for the 0-accuracy mammals
- **Hard negative mining**: the nocturnal confusion cluster suggests the model needs
  explicit negative examples to distinguish similar-sounding species

### 7.4 Site diversity is a major lever

The 14 new sites in unlabeled data represent acoustic environments the model has never
seen in labeled form. S01 (2,341 files) and S02 (2,505 files) alone are massive — together
they're 6× larger than all labeled data. Pseudo-labeling these gives the model exposure
to diverse acoustic environments, which is exactly what robustness requires.

### 7.5 The Perch+SED ensemble should be the teacher

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
