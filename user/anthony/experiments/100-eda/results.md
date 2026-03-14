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
| Labeled soundscapes | **1.03h** (66 files, 739 unique windows) |
| Unlabeled soundscapes | 176.5h (10,592 files) |
| Taxonomy entries | 234 |
| Submission targets | 234 columns (1:1 match with taxonomy) |
| Training species dirs | 206 |
| Sample rate | 32,000 Hz (all files) |
| Missing coordinates | 0 / 35,549 |

### Label formats

| Format | Species | Focal recordings | Example |
|--------|---------|------------------|---------|
| eBird code | 162 | 34,799 | `rubthr1`, `banana` |
| iNat numeric ID | 44 | 750 | `1161364`, `22973` |
| Song-type split | 25 | 0 | `47158son01`–`47158son25` |
| **No focal data at all** | 3 | 0 | `1491113`, `25073`, `517063` |

28 of 234 taxonomy entries have no `train_audio/` directory: 25 are `47158sonXX` song-type splits (parent species `47158` also has no directory), 3 are genuinely missing amphibians.

### Collections

| Collection | Recordings | Mean rating | Notes |
|------------|-----------|-------------|-------|
| XenoCanto | 23,043 | 4.01 | Aves (22,952), Amphibia (58), Mammalia (33) |
| iNaturalist | 12,506 | 0.0 | All 5 classes. Rating 0.0 = not rated, not low quality |

All Insecta (199) and Reptilia (1) recordings are iNat-only.

---

## 2. Class Imbalance

**Imbalance ratio: 499x** (max 499, min 1, median 125).

Class weights: inverse-frequency, mean-normalized to 1.0. Available in `class_weights.json`. **Note: covers 206 focal species only. The 28 non-focal species (25 song-types + 3 missing amphibians) have no weight and must be handled separately.**

### Focal training data by class

| Class | Species | Recordings | % total | Hours | Mean dur. | Median dur. |
|-------|---------|------------|---------|-------|-----------|-------------|
| Aves | 162 | 34,799 | 97.89% | 338.4h | 35.0s | 21.0s |
| Amphibia | 32 | 451 | 1.27% | 3.4h | 27.3s | 21.6s |
| Insecta | 3 | 199 | 0.56% | 1.6h | 29.2s | 21.1s |
| Mammalia | 8 | 99 | 0.28% | 1.0h | 38.1s | 14.9s |
| Reptilia | 1 | 1 | 0.003% | 0.002h | 7.9s | 7.9s |

### Focal duration quantiles

| p5 | p25 | p50 | p75 | p95 |
|----|-----|-----|-----|-----|
| 4.1s | 10.8s | 21.0s | 39.8s | 103.6s |

### Effective 5s training windows (bottom 10 species)

For cmAP on 5s soundscape windows, the effective training data per species is the number of 5s windows extractable from focal recordings, not the recording count.

| Species | Common name | Class | Files | 5s windows |
|---------|-------------|-------|-------|------------|
| 116570 | Southern Spectacled Caiman | Reptilia | 1 | **1** |
| 23724 | (Amphibia) | Amphibia | 1 | **1** |
| 23150 | (Amphibia) | Amphibia | 1 | 3 |
| 24321 | Mato Grosso Snouted Tree Frog | Amphibia | 2 | 4 |
| 209233 | (Mammalia) | Mammalia | 2 | 4 |
| 25214 | (Amphibia) | Amphibia | 3 | 7 |
| 70711 | (Amphibia) | Amphibia | 2 | 8 |
| 516975 | Hooded Capuchin | Mammalia | 1 | 8 |
| sptnig1 | Spot-tailed Nightjar | Aves | 3 | 10 |
| 476521 | (Insecta) | Insecta | 3 | 11 |

Two species have a single usable 5s window. At this level, data augmentation or external data is essential.

### Top 10 species by recordings

| Species | Common name | Class | Recordings |
|---------|-------------|-------|------------|
| rubthr1 | Rufous-bellied Thrush | Aves | 499 |
| banana | Bananaquit | Aves | 498 |
| fepowl | Ferruginous Pygmy-Owl | Aves | 497 |
| soulap1 | Southern Lapwing | Aves | 497 |
| houspa | House Sparrow | Aves | 496 |
| osprey | Osprey | Aves | 495 |
| coffal1 | Collared Forest-Falcon | Aves | 495 |
| socfly1 | Social Flycatcher | Aves | 494 |
| yeofly1 | Yellow-olive Flycatcher | Aves | 493 |
| compau | Pauraque | Aves | 493 |

### All non-bird species

**Insecta** (3 focal species, 199 recordings — all iNat):

| Species | Recordings | 5s windows |
|---------|------------|------------|
| 244024 | 181 | ~1260 |
| 1161364 | 11 | ~77 |
| 760266 | 7 | ~49 |

Note: 25 additional Insecta species (`47158son01`–`47158son25`) exist only in soundscape labels with zero focal data.

**Mammalia** (8 species, 99 recordings):

| Species | Common name | Recordings |
|---------|-------------|------------|
| 47144 | Domestic Dog | 43 |
| 41970 | (Mammalia) | 21 |
| 43435 | Black Howling Monkey | 14 |
| 74113 | Highland | 10 |
| 738183 | (Mammalia) | 5 |
| 74580 | (Mammalia) | 3 |
| 209233 | (Mammalia) | 2 |
| 516975 | Hooded Capuchin | 1 |

**Reptilia** (1 species, 1 recording):

| Species | Common name | Recordings | Duration |
|---------|-------------|------------|----------|
| 116570 | Southern Spectacled Caiman | 1 | 7.9s (1 usable 5s window) |

---

## 3. Secondary Labels

| Metric | Value |
|--------|-------|
| Recordings with secondary labels | 4,372 / 35,549 (12.3%) |
| Mean secondaries (when present) | 1.7 |
| Max secondaries | 15 |

### Distribution

| # secondary labels | Recordings |
|-------------------|------------|
| 0 | 31,177 |
| 1 | 2,623 |
| 2 | 1,049 |
| 3 | 379 |
| 4 | 169 |
| 5+ | 152 |

### By class

| Class | Total | With secondary | % |
|-------|-------|---------------|---|
| Aves | 34,799 | 4,358 | 12.5% |
| Amphibia | 451 | 13 | 2.9% |
| Mammalia | 99 | 1 | 1.0% |
| Insecta | 199 | 0 | 0% |
| Reptilia | 1 | 0 | 0% |

Secondary labels are almost exclusively a bird phenomenon (99.7% of secondaries are on Aves recordings). Non-bird taxa have essentially no multi-label annotations.

---

## 4. Soundscape Analysis

### Overview

| Metric | Value |
|--------|-------|
| Total soundscape files | 10,658 (177.6h, all 60s) |
| Labeled files | 66 (1.1h) across 9 sites |
| Unique labeled 5s windows | **739** |
| Unlabeled windows in labeled files | 26 (in 6 files) |
| Unlabeled files | 10,592 (176.5h) |
| Unique species in labels | 75 |
| Recording sites (all soundscapes) | 23 |

**Data note**: the raw `train_soundscapes_labels.csv` has 1,478 rows because every window is duplicated exactly once (2 identical rows per window). The true window count is 739.

### Labeled soundscape sites

| Site | Labeled files | Windows | Species | Total files (all soundscapes) |
|------|--------------|---------|---------|-------------------------------|
| S22 | 40 | 12 | 29 | 3,383 |
| S09 | 5 | 11 | 5 | 12 |
| S08 | 5 | 12 | 19 | 5 |
| S15 | 4 | 12 | 14 | 43 |
| S19 | 3 | 12 | 17 | 76 |
| S23 | 3 | 12 | 14 | 3 |
| S13 | 2 | 12 | 6 | 1,873 |
| S18 | 2 | 12 | 4 | 54 |
| S03 | 2 | 12 | 4 | 5 |

All soundscapes span 23 sites. The 3 largest unlabeled sites (S22: 3,383, S02: 2,505, S01: 2,341) account for 77% of all soundscape files. S22 also has the most labels (40 files).

### Taxa presence in labeled windows

| Taxa | Windows present | % of 739 | Species | Mean count/window |
|------|----------------|----------|---------|-------------------|
| Amphibia | 566 | 76.6% | 17 | 7.4 |
| Aves | 212 | 28.7% | 28 | 3.9 |
| Insecta | 168 | 22.7% | 25 | 6.8 |
| Mammalia | 41 | 5.5% | 4 | 2.0 |
| Reptilia | 13 | 1.8% | 1 | 2.0 |

Amphibians dominate Pantanal soundscapes (76.6%), not birds (28.7%). Multi-taxa windows: 240/739 (32.5%).

### Species density per window

| p25 | p50 | p75 | p90 | max |
|-----|-----|-----|-----|-----|
| 6 | 8 | 10 | 14 | 20 |

### Per-species soundscape window counts (all 75 species)

Most species appear in exactly 12 windows (the max across labeled files). Species with fewer windows have weaker soundscape supervision.

**12 windows** (42 species): 22961, 22967, 22973, 23158, 24279, 24321, 25073, 25092, 326272, 43435, 47158son01–son25 (except son02/05/09/12/19/21), 517063, 1491113, 555146, 65380, 66971, bufpar, chacha1, chvcon1, compau, litnig1, trsowl, undtin1, whtdov

**11 windows** (3): 47144 (Domestic Dog), 47158son21, nacnig1, orwpar

**6–10 windows** (7): 47158son02 (7), 47158son05 (3), 47158son09 (6), 47158son12 (5), 47158son19 (5), 116570 (9), 65377 (9), 67107 (9), bafcur1 (9), redjun (9), 516975 (8)

**1–5 windows** (14): compot1 (3), fusfly1 (3), magant1 (4), purjay1 (4), rufhor2 (4), 67252 (2), 74113 (2), bunibi1 (2), grekis (2), limpki (2), strher2 (2), thlwre1 (2), plcjay1 (1), rutjac1 (1), ruther1 (1), sibtan2 (1), wfwduc1 (1)

5 species have only 1 labeled window — essentially unlearnable from soundscape data alone.

### Species coverage: focal vs soundscape

| Class | Total species | In focal | In soundscape | In both |
|-------|--------------|----------|---------------|---------|
| Aves | 162 | 162 | 28 | 28 |
| Amphibia | 35 | 32 | 17 | 14 |
| Insecta | 28 | 3 | 25 | **0** |
| Mammalia | 8 | 8 | 4 | 4 |
| Reptilia | 1 | 1 | 1 | 1 |

**Zero overlap** between focal Insecta (3 species: 244024, 1161364, 760266) and soundscape Insecta (25 `47158sonXX` song-types). These are entirely separate species sets.

### Soundscape-only species (no focal training data)

| Species | Common name | Class | Windows | Files |
|---------|-------------|-------|---------|-------|
| 47158son01–25 | Insect sonotype 01–25 | Insecta | 3–12 each | 1–7 each |
| 1491113 | Guaraní Leaf-litter Frog | Amphibia | 12 | 8 |
| 25073 | Chiasmocleis mehelyi | Amphibia | 12 | 1 |
| 517063 | Southern Orange-legged Leaf Frog | Amphibia | 12 | 40 |

---

## 5. The 47158 Song-Type Problem

Species `47158` does **not exist** in `train_audio/` — there is no parent directory and no focal recordings. The taxonomy contains only the 25 song-type entries (`47158son01`–`47158son25`), each listed as "Insect sonotype NN" with the scientific name "Insect sonNN".

This means:
- **Zero focal training data** for any of the 25 song types
- All supervision comes from **soundscape labels only** (3–12 windows each)
- No metadata exists to distinguish song types — they are defined purely by soundscape annotation
- These 25 targets represent 10.7% of all 234 submission columns

### Song-type window counts

| Song type | Windows | Files |
|-----------|---------|-------|
| son01–son04, son06–08, son10–11, son13–18, son20, son22–25 | 12 | 1–7 |
| son21 | 11 | 2 |
| son02 | 7 | 1 |
| son09 | 6 | 1 |
| son05, son12, son19 | 3–5 | 1–2 |

---

## 6. Geographic Domain Gap

Test soundscapes: **Pantanal, Brazil** (lat -21.6 to -16.5, lon -57.6 to -55.9).

| Metric | Value |
|--------|-------|
| Focal recordings in Pantanal | 847 / 35,549 (2.4%) |
| Species with Pantanal recordings | 119 / 206 (57.8%) |
| Species with zero Pantanal recordings | 87 / 206 (42.2%) |

### By class

| Class | Species | In Pantanal | Pantanal recs |
|-------|---------|-------------|---------------|
| Aves | 162 | 110 (67.9%) | 834 |
| Amphibia | 32 | 5 (15.6%) | 6 |
| Mammalia | 8 | 4 (50.0%) | 7 |
| Insecta | 3 | 0 (0%) | 0 |
| Reptilia | 1 | 0 (0%) | 0 |

**Zero** Insecta or Reptilia focal recordings from the Pantanal. Amphibia has only 6 recordings from the region despite being the dominant taxon in soundscapes.

### Top 10 species by Pantanal fraction

| Species | Common name | Class | Total | Pantanal | % |
|---------|-------------|-------|-------|----------|---|
| magant1 | Mato Grosso Antbird | Aves | 63 | 40 | 63.5% |
| hyamac1 | Hyacinth Macaw | Aves | 65 | 39 | 60.0% |
| 738183 | (Mammalia) | Mammalia | 5 | 3 | 60.0% |
| 24321 | Mato Grosso Snouted Tree Frog | Amphibia | 2 | 1 | 50.0% |
| whlspi1 | White-lored Spinetail | Aves | 59 | 28 | 47.5% |
| 64898 | (Amphibia) | Amphibia | 3 | 1 | 33.3% |
| yebcar | Yellow-billed Cardinal | Aves | 27 | 8 | 29.6% |
| fabwre1 | Fawn-breasted Wren | Aves | 79 | 21 | 26.6% |
| purjay1 | Purplish Jay | Aves | 77 | 20 | 26.0% |
| bafcur1 | Bare-faced Curassow | Aves | 51 | 13 | 25.5% |

---

## 7. Species Overlap with BirdCLEF 2025

| Category | Count |
|----------|-------|
| In both 2025 and 2026 | 42 (18%) |
| New in 2026 | 192 |
| Dropped from 2025 | 164 |

### By class

| Class | In 2026 | Also in 2025 | % overlap |
|-------|---------|--------------|-----------|
| Aves | 162 | 39 | 24.1% |
| Amphibia | 35 | 2 | 5.7% |
| Mammalia | 8 | 1 | 12.5% |
| Insecta | 28 | 0 | 0% |
| Reptilia | 1 | 0 | 0% |

No Insecta or Reptilia overlap. Backbone features may transfer but heads need full retraining.

---

## 8. Test Set Structure

The sample submission is a **placeholder**: 3 windows from 1 file at 1 site (S05) on 2025-02-27, hour 01. The actual hidden test set will be larger and span more sites/dates.

---

## 9. Derived Artifacts

Regenerate via `sbatch sbatch/eda.sbatch`. Artifacts are gitignored.

| File | Description |
|------|-------------|
| `label_mapping.json` | primary_label → submission column index (234 targets) |
| `class_weights.json` | Inverse-frequency weights for 206 focal species. **Does not include 28 non-focal species** |
| `train_metadata.parquet` | train.csv + taxonomy join + n_secondary, has_location, label_format, class_name |
| `audio_durations.parquet` | Duration + sample rate for all 35,549 focal files |
| `class_distribution.parquet` | Per-species × collection counts + ratings |
| `taxonomy_coverage.parquet` | Per-species: has_train_audio, in_submission, is_songtype, parent_species |
| `soundscape_labels.parquet` | Exploded labels: 1 row per species per window (6,244 rows, deduplicated) |
| `soundscape_window_stats.parquet` | Per-window: species count, per-taxa counts, multi-taxa flag |
| `soundscape_taxa_summary.parquet` | Per-taxa presence rates and density |
| `soundscape_negatives.parquet` | Per labeled file: labeled vs unlabeled window counts |
| `soundscape_coverage.parquet` | Focal vs soundscape species overlap |
| `geographic_gap.parquet` | Per-species Pantanal bounding box overlap |
| `species_overlap_2025.parquet` | 2025↔2026 species comparison by scientific name |
| `test_set_structure.parquet` | Parsed submission row IDs → site/date/time |

---

## 10. Implications for Modeling

1. **Class weighting is essential.** 499x imbalance with cmAP means rare non-bird species drive the score. Use `class_weights.json` but add manual weights for the 28 non-focal species.

2. **Soundscape pseudo-labeling is the biggest opportunity.** 176.5h of unlabeled in-domain soundscapes vs 1h labeled. Teacher models (BirdNET, Perch) or self-training could provide orders of magnitude more in-domain supervision.

3. **Geographic domain gap is severe.** Only 2.4% of focal data is from the Pantanal. Zero Insecta/Reptilia focal recordings from the test region. Amphibia has only 6 Pantanal recordings despite dominating soundscapes (76.6% of windows). Domain adaptation or soundscape finetuning is critical.

4. **Amphibians dominate the test domain**, not birds. Models optimized for bird classification will underperform. The acoustic environment is wetland/nocturnal, not forest/diurnal.

5. **Song-type discrimination (47158) is the hardest subproblem.** 25 submission targets (10.7% of all targets) with zero focal data. All 25 are defined only by soundscape annotations (3–12 windows each). Options: predict uniform across all 25, use unsupervised clustering on soundscape windows, or treat as a retrieval problem.

6. **3 zero-shot amphibian species** (1491113, 25073, 517063) have no focal data. Only learnable from the 66 labeled soundscape files. Species 517063 appears in 40 soundscape files (the most of any species) but only 12 labeled windows.

7. **Secondary labels are bird-only.** 12.3% of recordings have secondaries (mean 1.7 when present), but 99.7% are on Aves. Non-bird multi-label training cannot rely on secondary annotations.

8. **Effective training data is much smaller than recording counts suggest.** Two species have a single usable 5s window. Per-species 5s window counts (not recording counts) should guide sampling strategy.

9. **2025 transfer is limited.** 18% species overlap (42/234). Backbone features may transfer but heads need retraining.

10. **Raw soundscape labels are duplicated.** Every window has exactly 2 identical rows in the CSV (1,478 rows → 739 unique windows). Pipelines must deduplicate or they will double-count.
