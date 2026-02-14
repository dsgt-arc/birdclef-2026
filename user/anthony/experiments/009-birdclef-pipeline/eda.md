# BirdCLEF 2025 Dataset — Exploratory Data Analysis

**Date:** 2026-02-13
**Source:** `~/scratch/birdclef-2026/data/birdclef-2025/`
**Script:** `birdclef-explore` (job 3951301)
**CSV outputs:** `logs/train_audio_durations.csv`, `logs/species_duration_stats.csv`, `logs/train_with_durations.csv`, `logs/train_soundscapes_durations.csv`

## Dataset Contents

```
birdclef-2025/
├── train.csv                 # 28,564 rows × 13 columns
├── taxonomy.csv              # 206 rows (species → scientific/common name)
├── sample_submission.csv     # 3 rows × 207 columns (row_id + 206 species)
├── recording_location.txt
├── train_audio/              # 28,564 focal recordings (.ogg)
├── train_soundscapes/        # 9,726 soundscape recordings (.ogg)
└── test_soundscapes/         # Empty (hidden at competition time)
```

## Metadata (`train.csv`)

| Field | Description |
|-------|-------------|
| `primary_label` | Species code (206 unique) |
| `secondary_labels` | Additional species in recording |
| `type` | Recording type |
| `filename` | Relative path: `{species_id}/{recording}.ogg` |
| `collection` | Source collection |
| `rating` | Quality rating |
| `url` | Source URL |
| `latitude`, `longitude` | Recording location |
| `scientific_name`, `common_name` | Species names |
| `author`, `license` | Attribution |

## Train Audio

| Metric | Value |
|--------|-------|
| Files | 28,564 |
| Format | OGG, 32 kHz |
| Total duration | 280.5 hours (1,009,808 s) |
| Estimated 10s chunks | 100,980 |

### Duration Distribution

| Bucket | Count | % |
|--------|------:|--:|
| 0–5 s | 2,314 | 8.1% |
| 5–10 s | 4,234 | 14.8% |
| 10–30 s | 11,683 | 40.9% |
| 30–60 s | 6,049 | 21.2% |
| 1–2 min | 3,057 | 10.7% |
| 2–5 min | 1,088 | 3.8% |
| 5–10 min | 119 | 0.4% |
| 10 min+ | 20 | 0.1% |

**Percentiles:** p5=3.9s, p25=10.6s, median=21.0s, p75=41.3s, p95=110.9s, max=1774.4s

Over 40% of recordings are 10–30 seconds. The median recording is 21 seconds, yielding ~2 full 10s chunks per file.

### Species — Recording Count

| Stat | Value |
|------|------:|
| Species | 206 |
| Min recordings | 2 |
| Max recordings | 990 |
| Median | 80.5 |
| Mean | 138.7 |

**Most recordings:**

| Species | Count |
|---------|------:|
| `grekis` | 990 |
| `compau` | 808 |
| `trokin` | 787 |
| `roahaw` | 709 |
| `banana` | 610 |
| `whtdov` | 572 |
| `socfly1` | 543 |
| `yeofly1` | 525 |
| `bobfly1` | 514 |
| `wbwwre1` | 499 |

**Fewest recordings:**

| Species | Count |
|---------|------:|
| `42113` | 2 |
| `47067` | 2 |
| `476537` | 2 |
| `528041` | 2 |
| `64862` | 2 |
| `66016` | 2 |
| `66531` | 2 |
| `66578` | 2 |
| `67082` | 2 |
| `81930` | 2 |

Note: Species with numeric-only labels (e.g. `42113`) are iNaturalist taxon IDs, not eBird codes. These tend to have very few recordings.

### Species — Total Audio Duration

| Stat | Value |
|------|------:|
| Min total | 11 s (0.2 min) |
| Max total | 32,493 s (9.0 h) |
| Median total | 2,637 s (44.0 min) |
| Mean total | 4,902 s (1.4 h) |

**Most audio:**

| Species | Duration | Recordings |
|---------|----------|----------:|
| `compau` | 9.0 h | 808 |
| `grekis` | 8.3 h | 990 |
| `roahaw` | 7.8 h | 709 |
| `whtdov` | 7.3 h | 572 |
| `laufal1` | 7.1 h | 467 |
| `trokin` | 6.9 h | 787 |
| `yercac1` | 5.8 h | 302 |
| `trsowl` | 5.7 h | 470 |
| `banana` | 5.6 h | 610 |
| `wbwwre1` | 5.3 h | 499 |

**Least audio:**

| Species | Duration | Recordings |
|---------|----------|----------:|
| `1564122` | 11 s | 6 |
| `21116` | 13 s | 2 |
| `42087` | 21 s | 2 |
| `42113` | 23 s | 2 |
| `868458` | 24 s | 4 |
| `523060` | 24 s | 2 |
| `66016` | 27 s | 2 |
| `548639` | 29 s | 2 |
| `67082` | 44 s | 2 |
| `81930` | 44 s | 2 |

The long tail is extreme: the top species has 2,950× more audio than the bottom. Ten species have fewer than 45 seconds total. This class imbalance is a key modeling challenge.

## Train Soundscapes

| Metric | Value |
|--------|-------|
| Files | 9,726 |
| Format | OGG, 32 kHz |
| Duration per file | 60.0 s (all identical) |
| Total duration | 162.1 hours |
| Estimated 10s chunks | 58,356 |

All soundscapes are exactly 60 seconds. These are the primary pretraining corpus for MSPM self-supervised learning.

## Sample Submission

| Metric | Value |
|--------|-------|
| Rows | 3 |
| Columns | 207 (1 `row_id` + 206 species) |
| Row ID format | `soundscape_{id}_{end_seconds}` |
| Example | `soundscape_8358733_5`, `soundscape_8358733_10`, `soundscape_8358733_15` |

Submission requires predictions at 5-second intervals within each soundscape. Each 10s model chunk produces two 5-second prediction windows.

## Taxonomy

206 species with columns: `primary_label`, `inat_taxon_id`, `scientific_name`, `common_name`, `class_name`.

## Grand Total

| Metric | Value |
|--------|-------|
| Total audio | **442.6 hours** |
| Total 10s chunks | **159,336** |

Breakdown: 280.5 h train audio + 162.1 h train soundscapes.

## Implications for Pipeline

1. **Pretraining corpus**: 162 hours of unlabeled soundscapes = 58k 10s chunks. Small compared to AudioSet but sufficient for initial MSPM experiments.
2. **Class imbalance**: 495:1 ratio in recording count, 2,950:1 in total audio. Focal loss and/or oversampling needed.
3. **Short recordings**: 23% of train audio is under 10 seconds. The data loader must handle padding for clips shorter than 10s.
4. **Uniform sample rate**: All files are 32 kHz — no resampling needed.
5. **Submission format**: 5-second windows from 10s chunks. SED token outputs (1000 tokens/10s = 100 tokens/s) naturally support this split at token 500.

## CSV Outputs

All CSVs are joinable for downstream analysis:

- **`train_audio_durations.csv`** (28,564 rows): `filename, duration_s, sample_rate, frames` — join with `train.csv` on `filename`
- **`species_duration_stats.csv`** (206 rows): `primary_label, num_recordings, total_duration_s, mean_duration_s, min_duration_s, max_duration_s, median_duration_s`
- **`train_with_durations.csv`** (28,564 rows): Full `train.csv` merged with duration info
- **`train_soundscapes_durations.csv`** (9,726 rows): `filename, duration_s, sample_rate, frames`
