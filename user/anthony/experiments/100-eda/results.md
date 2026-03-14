# Results: BirdCLEF 2026 Exploratory Data Analysis

**Status**: completed
**Date**: 2026-03-14
**Author**: Anthony

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total audio | **522.1 hours** across 46,207 files |
| Focal recordings | 344.5h, 35,549 files, mean 34.9s, median 21.0s |
| Soundscape recordings | 177.6h, 10,658 files, all exactly 60s |
| Taxonomy entries | 234 (162 Aves, 35 Amphibia, 28 Insecta, 8 Mammalia, 1 Reptilia) |
| Submission targets | 234 columns (1:1 match with taxonomy) |
| Training species dirs | 206 (28 taxonomy entries lack their own directory) |
| Collections | XenoCanto: 23,043 (mean rating 4.0), iNaturalist: 12,506 (rating 0.0) |
| Missing coordinates | 0 / 35,549 (all have lat/lon) |

## Key Finding 1: Extreme Class Imbalance

**Imbalance ratio: 499x** (max 499 recordings, min 1 recording, median 125).

### Per-class breakdown

| Class | Species | Recordings | Hours | Mean duration |
|-------|---------|------------|-------|---------------|
| Aves | 162 | 34,799 (97.9%) | 338.4h | 35.0s |
| Amphibia | 32 | 451 (1.3%) | 3.4h | 27.3s |
| Insecta | 3 | 199 (0.6%) | 1.6h | 29.2s |
| Mammalia | 8 | 99 (0.3%) | 1.0h | 38.1s |
| Reptilia | 1 | 1 (0.003%) | 0.002h | 7.9s |

Birds dominate the focal training data (97.9% of recordings). Non-bird taxa have extremely limited training data: 3 insect species with 199 recordings, 1 reptile with a single 8-second clip. For cmAP (class-mean AP), every species matters equally — the model must perform well on these underrepresented taxa.

### Most / least represented species

**Top 5**: rubthr1 (499), banana (498), fepowl (497), soulap1 (497), houspa (496) — all Aves

**Bottom 5**: 23724 (1, Amphibia), 23150 (1, Amphibia), 516975 (1, Mammalia), 116570 (1, Reptilia), 209233 (2, Mammalia)

## Key Finding 2: Labeled Soundscapes Are Sparse but Dense

Only **66 of 10,658 soundscape files** (0.6%) have labels — just **1.03 hours** of labeled soundscape audio out of 177.6 hours total.

However, labeled windows are **extremely dense**:

| Metric | Value |
|--------|-------|
| Labeled windows | 739 (out of 765 possible in 66 files = 96.6% coverage) |
| Candidate negative windows | 26 (too few for hard negative mining) |
| Species per window | median 8, p90 14, max 20 |
| Multi-taxa windows | 240 / 739 (32.5%) |

### Taxa presence in soundscape windows

| Taxa | Windows present | % of 739 | Unique species |
|------|----------------|----------|----------------|
| Amphibia | 566 | 76.6% | 17 |
| Aves | 212 | 28.7% | 28 |
| Insecta | 168 | 22.7% | 25 |
| Mammalia | 41 | 5.5% | 4 |
| Reptilia | 13 | 1.8% | 1 |

**Amphibians dominate soundscapes (76.6%)**, not birds (28.7%). The test site (Pantanal wetland) is a very different acoustic environment from the bird-heavy focal training data.

### Soundscape species coverage

| Category | Count | Notes |
|----------|-------|-------|
| Both focal + soundscape | 47 | Core overlap |
| Focal only | 159 | No soundscape supervision |
| Soundscape only | 28 | 25 are 47158 song-type splits, 3 are genuinely soundscape-only species |
| Neither | 0 | |

The 28 "soundscape-only" species are almost entirely the 47158 song-type splits (25 Insecta sub-labels). The 3 genuinely soundscape-only species are 1491113, 25073, and 517063 (all Amphibia) — these have **zero focal training data** and can only be learned from soundscape labels.

## Key Finding 3: Massive Geographic Domain Gap

Test soundscapes are from the **Pantanal, Brazil** (lat -21.6 to -16.5, lon -57.6 to -55.9).

| Metric | Value |
|--------|-------|
| Focal recordings in Pantanal | 847 / 35,549 (2.4%) |
| Species with any Pantanal recordings | 119 / 206 (57.8%) |
| Species with zero Pantanal recordings | 87 / 206 (42.2%) |

Only 2.4% of focal training data comes from the test region. For 87 species, the model has never seen a recording from the Pantanal. Domain adaptation (geographic, acoustic environment) is likely critical.

## Key Finding 4: Limited Transfer from BirdCLEF 2025

| Category | Count |
|----------|-------|
| Species in both 2025 and 2026 | 42 (18%) |
| New in 2026 | 192 |
| Dropped from 2025 | 164 |

Only 42 species carry over. Pre-trained models from BirdCLEF 2025 have limited direct transfer value — backbone features may generalize, but species-level heads need retraining from scratch.

## Key Finding 5: The 47158 Song-Type Problem

Species 47158 (Insecta) has a single `train_audio/47158/` directory but **25 submission columns** (`47158son01` through `47158son25`). All 25 song-type labels appear only in soundscape labels — there is no metadata mapping focal recordings to song types. This means:

- The 25 song types cannot be trained from focal data without external annotation
- All 25 are "soundscape-only" targets
- A model must learn fine-grained insect song-type discrimination from only soundscape windows

## Key Finding 6: Missing Species

Three species have **zero training audio** (no focal recordings, no `train_audio/` directory):

| Species | Class | In soundscape labels? |
|---------|-------|----------------------|
| 1491113 | Amphibia | Yes |
| 25073 | Amphibia | Yes |
| 517063 | Amphibia | Yes |

All three appear in soundscape labels, so they can potentially be learned from the 66 labeled soundscape files. But with only ~1 hour of labeled soundscape audio total, the supervision signal is minimal.

## Derived Artifacts

All artifacts are in `artifacts/` (1.2MB total, checked into repo):

| File | Size | Description |
|------|------|-------------|
| `label_mapping.json` | 8K | primary_label → submission column index (234 targets) |
| `class_weights.json` | 5K | Inverse-frequency weights for cmAP training |
| `train_metadata.parquet` | 848K | Enriched train.csv with taxonomy + derived columns |
| `audio_durations.parquet` | 288K | Duration + sample rate for all 35,549 focal files |
| `soundscape_labels.parquet` | 18K | Exploded soundscape labels (6,244 rows) |
| `soundscape_window_stats.parquet` | 5K | Per-window species density stats |
| `soundscape_taxa_summary.parquet` | 2K | Per-taxa soundscape presence summary |
| `soundscape_negatives.parquet` | 3K | Candidate negative windows (26 total) |
| `soundscape_coverage.parquet` | 4K | Focal vs soundscape species overlap |
| `class_distribution.parquet` | 12K | Per-species × collection recording counts |
| `taxonomy_coverage.parquet` | 11K | Per-species training data availability |
| `geographic_gap.parquet` | 9K | Per-species Pantanal overlap |
| `species_overlap_2025.parquet` | 9K | 2025↔2026 species comparison |
| `test_set_structure.parquet` | 3K | Parsed test submission row IDs |

## Implications for Modeling

1. **Class weighting is essential.** 499x imbalance with cmAP metric means rare species (especially non-birds) drive the score. Use the provided `class_weights.json`.

2. **Domain adaptation matters more than architecture.** 97.6% of training data is outside the Pantanal. Soundscape finetuning, geographic augmentation, or pseudo-labeling on unlabeled soundscapes could help close this gap.

3. **Soundscape pseudo-labeling is the biggest opportunity.** 176.5 hours of unlabeled soundscapes from the test domain — teacher models (BirdNET, Perch) or self-training could provide orders of magnitude more in-domain supervision than the 1 hour of labeled soundscape data.

4. **Multi-taxa mel configuration.** Insects (broadband stridulation), amphibians (low-frequency calls), and birds (harmonic 1-10kHz) have different spectral characteristics. A single mel config may waste resolution. Consider taxon-aware frequency ranges or multi-scale spectrograms.

5. **Song-type discrimination requires creative solutions.** The 25 `47158sonXX` targets can only be learned from soundscape windows. Unsupervised clustering of focal recordings in `47158/`, external metadata, or treating them as a single class with post-hoc splitting may be needed.

6. **Zero-shot species exist.** Three amphibian species have no focal training data. Models must generalize to these from soundscape context alone, or use external data sources.

7. **2025 transfer is limited.** Only 18% species overlap — retrain heads, but backbone features may still transfer.
