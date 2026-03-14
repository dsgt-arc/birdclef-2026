---
title: "BirdCLEF 2026 Competition Analysis"
source: "https://www.kaggle.com/competitions/birdclef-2026"
date_captured: "2026-03-13"
data_path: "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/"
---

# BirdCLEF 2026 Competition Analysis

**Competition**: [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)
**Data page**: [kaggle.com/competitions/birdclef-2026/data](https://www.kaggle.com/competitions/birdclef-2026/data)
**Local data**: `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/`

## Task

Identify which species (birds, amphibians, mammals, reptiles, insects) are calling in 1-minute soundscape recordings from the Brazilian Pantanal. For each 5-second segment of each test soundscape, predict the probability of presence for each of 234 species. Hidden test set (~600 soundscapes) scored via submitted Kaggle notebook.

## 2025 vs 2026 Comparison

| | BirdCLEF 2025 | BirdCLEF 2026 |
|---|---|---|
| **Location** | El Silencio Reserve, Colombia | Pantanal, Brazil |
| **Taxa** | Aves, Amphibia, Mammalia, Insecta | Aves, Amphibia, Mammalia, Insecta, **+ Reptilia** |
| **Submission species** | 206 | **234** (+14%) |
| **Species in train_audio** | 206 | 206 |
| **Species ONLY in soundscapes** | 0 | **28** (new problem) |
| **Train audio files** | 28,564 | **35,549** (+24%) |
| **Collections** | XC + iNat + CSA | XC + iNat (no CSA) |
| **Train soundscapes** | 9,726 (all unlabeled) | **10,658 (66 labeled + 10,592 unlabeled)** |
| **Soundscape labels** | None | **1,478 expert-annotated 5s segments** |
| **Sample rate** | 32 kHz | 32 kHz |
| **Test soundscapes** | ~700 hidden, 1 min | ~600 hidden, 1 min |
| **Max recordings/species** | 990 (grekis) | 499 (rubthr1) |
| **Min recordings/species** | 2 | **1** |

## What's New in 2026

### 1. Expert-labeled soundscapes (game-changer)

66 labeled soundscape files with 1,478 annotated 5-second segments. BirdCLEF 2025 had zero soundscape labels. This provides:

- Real multi-label ground truth from the target domain (avg 4.2 species per segment)
- Supervision for 28 species that have no focal recordings at all
- A validation signal for soundscape performance before submitting

### 2. Soundscape-only species

234 submission species but only 206 in train_audio. The missing 28 have no focal recordings — they exist only in the labeled soundscapes:

- **25 insect sonotypes** (47158son01 through 47158son25) — unnamed insect sounds identified only by acoustic pattern
- **2 amphibians** — Guaraní leaf-litter frog (1491113), Chiasmocleis mehelyi (25073)
- **1 amphibian** — Southern Orange-legged Leaf Frog (517063)

You cannot build a classifier for these 28 species from focal recordings alone.

### 3. Reptilia

New taxonomic class — 1 recording of a Southern Spectacled Caiman (116570). Minor addition but new.

### 4. Different geography

Brazilian Pantanal wetlands vs. Colombian cloud forest. Different acoustic environment, species assemblage, and background noise characteristics.

### 5. Soundscape filename metadata

Filenames follow `BC2026_Train_<ID>_<site>_<date>_<time in UTC>.ogg`, encoding recording site, date, and time of day. BirdCLEF 2025 soundscape filenames did not encode this metadata. Enables spatial and temporal features (nocturnal frogs vs diurnal birds, site-specific species pools).

## Data Structure

### train_audio/ — 35,549 focal recordings, 206 species

Organized in subdirectories by species label. Filenames: `[collection][file_id].ogg`. Resampled to 32 kHz, ogg format. Sources: Xeno-Canto (23,043) and iNaturalist (12,506).

| Class | Recordings | Species | Notes |
|---|---|---|---|
| Aves | 34,799 (98%) | 162 | Dominates training data |
| Amphibia | 451 (1.3%) | 27 | Frogs — many Pantanal species |
| Insecta | 199 (0.6%) | 3 (named) | Giant Cicada, Guyalna cuta, Prionacris erosa |
| Mammalia | 99 (0.3%) | 7 | Jaguar, monkeys, domestic animals |
| Reptilia | 1 (<0.01%) | 1 | Southern Spectacled Caiman |

Class imbalance is severe:
- Top: 499 recordings (rubthr1 — Rufous-bellied Thrush)
- Bottom: 1 recording (516975 — Hooded Capuchin)
- 4 species with 2-3 recordings, 28 species with zero focal recordings

### train_soundscapes/ — 10,658 files from 23 recording sites

1-minute soundscape recordings from the Pantanal. Same general recording locations as hidden test data, but precise dates/times do not overlap.

- **66 labeled** (0.6%) → 1,478 annotated 5-second segments in `train_soundscapes_labels.csv`
- **10,592 unlabeled** (99.4%)

Site distribution (highly uneven):

| Site | Files | % of Total |
|---|---|---|
| S22 | 3,383 | 31.7% |
| S02 | 2,505 | 23.5% |
| S01 | 2,341 | 22.0% |
| S13 | 1,873 | 17.6% |
| All other 19 sites | 556 | 5.2% |

### train_soundscapes_labels.csv — Expert annotations

Columns: `filename`, `start`, `end`, `primary_label` (semicolon-separated multi-label).

Key statistics:
- 1,478 labeled 5-second segments across 66 soundscape files
- **75 unique species** annotated (out of 234 submission species)
- Average **4.2 species per segment** (densely multi-label)
- Maximum: 10 species in a single 5s segment

Labels per segment distribution:

| Species/segment | Segments | % |
|---|---|---|
| 1 | 156 | 10.6% |
| 2 | 98 | 6.6% |
| 3 | 200 | 13.5% |
| 4 | 364 | 24.6% |
| 5 | 320 | 21.6% |
| 6 | 190 | 12.9% |
| 7 | 114 | 7.7% |
| 8+ | 36 | 2.4% |

Species breakdown in labeled soundscapes:
- Aves: 28 species
- Insecta: 25 species (all sonotypes — soundscape-only)
- Amphibia: 17 species
- Mammalia: 4 species
- Reptilia: 1 species

Most frequent in labeled soundscapes: Dendropsophus nanus (65380, 666 segments), Southern Orange-legged Leaf Frog (517063, 626), Whistling Grass Frog (22973, 426).

### train.csv — Focal recording metadata

Columns: `primary_label`, `secondary_labels`, `type`, `latitude`, `longitude`, `scientific_name`, `common_name`, `class_name`, `inat_taxon_id`, `author`, `license`, `rating`, `url`, `filename`, `collection`.

- 12.3% of recordings have secondary labels (background species)
- Ratings: mean 2.60 (scale 1-5), 12,849 recordings have no rating (iNat doesn't provide ratings)
- Coordinates: lat -16.5 to -21.6, lon -55.9 to -57.6 (Pantanal region)

### taxonomy.csv — Species metadata

234 rows (= 234 submission columns). Columns: `primary_label`, `inat_taxon_id`, `scientific_name`, `common_name`, `class_name`.

Notable: insect sonotypes (47158son01–25) share iNat taxon ID 47158 (Insecta) but are treated as separate classes. They have no species-level identification.

### sample_submission.csv — Submission format

Row ID format: `BC2026_Test_<file_ID>_<site>_<date>_<time>_<end_seconds>`. 234 species probability columns. Default value: 1/234 ≈ 0.00427.

Test soundscapes are 1 minute each, split into 12 × 5-second segments → 12 rows per file.

## Key Implications for Our Approach

### 1. Labeled soundscapes are mandatory

Without them, 28/234 = 12% of species cannot be predicted at all. The 25 insect sonotypes have no focal recordings and exist only in the expert-annotated soundscape segments.

### 2. MixIT separation is highly relevant

Labeled soundscape segments average 4.2 overlapping species per 5s segment. Sound separation could:
- Isolate individual species from dense mixtures for cleaner training
- Improve pseudo-labeling of the 10,592 unlabeled soundscape files
- Boost classifier confidence on low-SNR vocalizations in test soundscapes

### 3. Multi-taxa challenge

Insect sonotypes are the hardest category: unnamed, no focal recordings, only appear in dense soundscape mixtures. Models pretrained on bird audio (EAT, ProtoCLR) may not transfer well to insects and frogs. ConvNeXt (ImageNet pretrained on spectrograms) may be more domain-agnostic.

### 4. Site/time metadata enables priors

Soundscape filenames give site ID, date, and UTC time. This enables:
- Time-of-day priors (nocturnal frogs/insects vs diurnal birds)
- Site-specific species pools (not all species occur at all sites)
- Seasonal patterns (dates range across months)

### 5. Our BirdCLEF 2025 models need retraining

Different species set, different location, different species assemblage. The pretrained encoders (EAT on AudioSet, ConvNeXt on ImageNet, ProtoCLR on Xeno-Canto) still provide strong frozen representations, but classification heads and LoRA adapters must be retrained for BirdCLEF 2026 species. The LoRA fine-tuning pipeline from experiments 006-009 transfers directly.
