# Results: BirdCLEF 2026 Exploratory Data Analysis

**Status**: completed
**Date**: 2026-03-14
**Author**: Anthony
**Artifacts**: `artifacts/` (1.2MB, regenerate via `sbatch sbatch/eda.sbatch`)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total audio | **522.1 hours** across 46,207 files |
| Focal recordings | 344.5h, 35,549 files, mean 34.9s, median 21.0s |
| Soundscape recordings | 177.6h, 10,658 files, all exactly 60.0s |
| Labeled soundscapes | **1.03h** (66 files, 739 windows — 0.6% of soundscapes) |
| Unlabeled soundscapes | 176.5h (10,592 files) |
| Taxonomy entries | 234 |
| Submission targets | 234 columns (1:1 match with taxonomy) |
| Training species dirs | 206 |
| Sample rate | 32,000 Hz (all files) |
| All recordings have lat/lon | Yes (0 missing) |

### Label format

| Format | Species | Recordings | Example |
|--------|---------|------------|---------|
| eBird code | 162 | 34,799 | `rubthr1`, `banana` |
| iNat numeric ID | 44 | 750 | `1161364`, `22973` |
| Song-type split | 25 | 0 (soundscape only) | `47158son01`–`47158son25` |

### Collection source

| Collection | Recordings | Mean rating | Classes covered |
|------------|-----------|-------------|-----------------|
| XenoCanto | 23,043 | 4.0 | Aves, Amphibia, Mammalia |
| iNaturalist | 12,506 | 0.0 | All 5 classes |

All Insecta and Reptilia recordings are iNat-only. iNat recordings have no quality rating.

---

## 2. Taxonomic Breakdown

### Focal training data by class

| Class | Species | Recordings | % of total | Hours | Mean dur. | Median dur. |
|-------|---------|------------|-----------|-------|-----------|-------------|
| Aves | 162 | 34,799 | 97.89% | 338.4h | 35.0s | 21.0s |
| Amphibia | 32 | 451 | 1.27% | 3.4h | 27.3s | 21.6s |
| Insecta | 3 | 199 | 0.56% | 1.6h | 29.2s | 21.1s |
| Mammalia | 8 | 99 | 0.28% | 1.0h | 38.1s | 14.9s |
| Reptilia | 1 | 1 | 0.003% | 0.002h | 7.9s | 7.9s |

### Focal duration quantiles (all files)

| p5 | p25 | p50 | p75 | p95 |
|----|-----|-----|-----|-----|
| 4.1s | 10.8s | 21.0s | 39.8s | 103.6s |

### Collection x class detail

| Class | XC | iNat | XC mean rating |
|-------|-----|------|----------------|
| Aves | 22,952 | 11,847 | 4.01 |
| Amphibia | 58 | 393 | 4.22 |
| Mammalia | 33 | 66 | 3.59 |
| Insecta | 0 | 199 | — |
| Reptilia | 0 | 1 | — |

---

## 3. Class Imbalance

**Imbalance ratio: 499x** (max 499, min 1, median 125)

### Top 10 species (most recordings)

| Species | Class | Recordings |
|---------|-------|------------|
| rubthr1 | Aves | 499 |
| banana | Aves | 498 |
| fepowl | Aves | 497 |
| soulap1 | Aves | 497 |
| houspa | Aves | 496 |
| osprey | Aves | 495 |
| coffal1 | Aves | 495 |
| socfly1 | Aves | 494 |
| yeofly1 | Aves | 493 |
| compau | Aves | 493 |

### Bottom 10 species (fewest recordings)

| Species | Class | Recordings |
|---------|-------|------------|
| 116570 | Reptilia | 1 |
| 516975 | Mammalia | 1 |
| 23150 | Amphibia | 1 |
| 23724 | Amphibia | 1 |
| 209233 | Mammalia | 2 |
| 24321 | Amphibia | 2 |
| 70711 | Amphibia | 2 |
| 555123 | Insecta | 3 (label name — actually mapped to species with 3 recs) |
| 25214 | Amphibia | 3 |
| 74580 | Mammalia | 3 |

### All non-bird species

**Insecta** (3 species, 199 recordings):

| Species | Recordings |
|---------|------------|
| 244024 | 181 |
| 1161364 | 11 |
| 760266 | 7 |

**Mammalia** (8 species, 99 recordings):

| Species | Recordings |
|---------|------------|
| 47144 | 43 |
| 41970 | 21 |
| 43435 | 14 |
| 74113 | 10 |
| 738183 | 5 |
| 74580 | 3 |
| 209233 | 2 |
| 516975 | 1 |

**Reptilia** (1 species, 1 recording):

| Species | Recordings |
|---------|------------|
| 116570 | 1 |

### Class weights (inverse-frequency, mean=1.0)

| Metric | Value |
|--------|-------|
| Min weight (most common) | 0.0344 (rubthr1, 499 recs) |
| Max weight (rarest) | 17.1786 (116570/516975/23150/23724, 1 rec each) |
| Median weight | ~1.0 |

---

## 4. Soundscape Analysis

### Overview

| Metric | Value |
|--------|-------|
| Total soundscape files | 10,658 (177.6h, all 60s) |
| Labeled files | 66 (1.1h) |
| Labeled 5s windows | 739 |
| Unlabeled windows in labeled files | 26 (in 6 files) |
| Unlabeled files | 10,592 (176.5h) |
| Unique species in labels | 75 |

### Taxa presence in labeled windows

| Taxa | Windows present | % of 739 | Unique species | Mean count when present |
|------|----------------|----------|----------------|------------------------|
| Amphibia | 566 | 76.6% | 17 | 7.4 |
| Aves | 212 | 28.7% | 28 | 3.9 |
| Insecta | 168 | 22.7% | 25 | 6.8 |
| Mammalia | 41 | 5.5% | 4 | 2.0 |
| Reptilia | 13 | 1.8% | 1 | 2.0 |

Amphibians dominate Pantanal soundscapes (76.6% of windows), not birds (28.7%).

### Species density per window

| p25 | p50 | p75 | p90 | max |
|-----|-----|-----|-----|-----|
| 6 | 8 | 10 | 14 | 20 |

Multi-taxa windows: 240 / 739 (32.5%)

### Species coverage: focal vs soundscape

| Class | Total species | In focal | In soundscape | In both |
|-------|--------------|----------|---------------|---------|
| Aves | 162 | 162 | 28 | 28 |
| Amphibia | 35 | 32 | 17 | 14 |
| Insecta | 28 | 3 | 25 | 0 |
| Mammalia | 8 | 8 | 4 | 4 |
| Reptilia | 1 | 1 | 1 | 1 |

Key: all 25 Insecta soundscape species are `47158sonXX` song-type splits with **zero overlap** with focal training data.

### Soundscape-only species (28 total)

- **25 song-type splits**: `47158son01`–`47158son25` (Insecta) — share `train_audio/47158/` but no metadata mapping
- **3 genuinely missing**: 1491113, 25073, 517063 (all Amphibia) — zero focal recordings, only learnable from soundscape labels

---

## 5. Geographic Domain Gap

Test soundscapes: **Pantanal, Brazil** (lat -21.6 to -16.5, lon -57.6 to -55.9)

| Metric | Value |
|--------|-------|
| Focal recordings in Pantanal | 847 / 35,549 (2.4%) |
| Species with Pantanal recordings | 119 / 206 (57.8%) |
| Species with zero Pantanal recordings | 87 / 206 (42.2%) |

### Pantanal coverage by class

| Class | Species | In Pantanal | Pantanal recordings |
|-------|---------|-------------|---------------------|
| Aves | 162 | 110 (67.9%) | 834 |
| Amphibia | 32 | 5 (15.6%) | 6 |
| Mammalia | 8 | 4 (50.0%) | 7 |
| Insecta | 3 | 0 (0%) | 0 |
| Reptilia | 1 | 0 (0%) | 0 |

Zero Insecta or Reptilia focal recordings from the Pantanal.

### Top species by Pantanal fraction

| Species | Class | Total recs | Pantanal recs | % Pantanal |
|---------|-------|------------|---------------|-----------|
| magant1 | Aves | 63 | 40 | 63.5% |
| hyamac1 | Aves | 65 | 39 | 60.0% |
| 738183 | Mammalia | 5 | 3 | 60.0% |
| 24321 | Amphibia | 2 | 1 | 50.0% |
| whlspi1 | Aves | 59 | 28 | 47.5% |
| 64898 | Amphibia | 3 | 1 | 33.3% |
| yebcar | Aves | 27 | 8 | 29.6% |
| fabwre1 | Aves | 79 | 21 | 26.6% |
| purjay1 | Aves | 77 | 20 | 26.0% |
| bafcur1 | Aves | 51 | 13 | 25.5% |

---

## 6. Species Overlap with BirdCLEF 2025

| Category | Count |
|----------|-------|
| In both 2025 and 2026 | 42 (18%) |
| New in 2026 | 192 |
| Dropped from 2025 | 164 |

### Overlap by class

| Class | Species in 2026 | Also in 2025 | % overlap |
|-------|----------------|--------------|-----------|
| Aves | 162 | 39 | 24.1% |
| Amphibia | 35 | 2 | 5.7% |
| Insecta | 28 | 0 | 0% |
| Mammalia | 8 | 1 | 12.5% |
| Reptilia | 1 | 0 | 0% |

No Insecta or Reptilia overlap. Backbone features may transfer but species heads need full retraining.

---

## 7. Test Set Structure

Sample submission has only **3 windows** from **1 file** at **1 site** (S05) on 1 date (2025-02-27, hour 01). This is a placeholder — the real hidden test set will be larger.

---

## 8. Derived Artifacts

Regenerate all via `sbatch sbatch/eda.sbatch`. Artifacts are gitignored.

| File | Description |
|------|-------------|
| `label_mapping.json` | primary_label → submission column index (234 targets) |
| `class_weights.json` | Inverse-frequency weights, imbalance stats |
| `train_metadata.parquet` | train.csv + taxonomy join + derived columns (35,549 rows) |
| `audio_durations.parquet` | Duration + sample rate for all focal files |
| `class_distribution.parquet` | Per-species × collection counts + ratings |
| `taxonomy_coverage.parquet` | Per-species: has_train_audio, in_submission, is_songtype |
| `soundscape_labels.parquet` | Exploded labels: 1 row per species per 5s window (6,244 rows) |
| `soundscape_window_stats.parquet` | Per-window: species count, per-taxa counts |
| `soundscape_taxa_summary.parquet` | Per-taxa presence summary |
| `soundscape_negatives.parquet` | Per labeled file: labeled vs unlabeled window counts |
| `soundscape_coverage.parquet` | Focal vs soundscape species overlap |
| `geographic_gap.parquet` | Per-species Pantanal bounding box overlap |
| `species_overlap_2025.parquet` | 2025↔2026 species comparison by scientific name |
| `test_set_structure.parquet` | Parsed submission row IDs → site/date/time |

---

## 9. Implications for Modeling

1. **Class weighting is essential.** 499x imbalance with cmAP means rare non-bird species drive the score. Use `class_weights.json`.

2. **Soundscape pseudo-labeling is the biggest opportunity.** 176.5h of unlabeled in-domain soundscapes vs 1h labeled. Teacher models (BirdNET, Perch) or self-training on unlabeled soundscapes could provide orders of magnitude more in-domain signal.

3. **Geographic domain gap is severe.** Only 2.4% of focal data is from the Pantanal. Zero Insecta/Reptilia focal recordings from the test region. Domain adaptation or soundscape finetuning is likely critical.

4. **Amphibians dominate the test domain.** Pantanal soundscapes are 76.6% amphibian, not bird-dominated. Models optimized for bird classification will underperform.

5. **Song-type discrimination (47158) requires creative solutions.** 25 submission targets with no focal training data and no type-assignment metadata. Options: unsupervised clustering of `47158/` recordings, external metadata, or predict uniform across all 25.

6. **3 zero-shot amphibian species exist.** 1491113, 25073, 517063 have no focal data — only learnable from the 66 labeled soundscape files.

7. **2025 transfer is limited.** 18% species overlap (42/234). Backbone features may transfer, heads need retraining.

8. **Multi-taxa mel config.** Different taxa have fundamentally different spectral signatures. A single mel configuration may waste resolution on non-bird taxa.
