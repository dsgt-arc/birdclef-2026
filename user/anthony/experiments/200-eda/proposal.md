# Proposal: BirdCLEF 2026 Exploratory Data Analysis

**Status**: proposed
**Created**: 2026-03-14
**Author**: Anthony

## Hypothesis

Understanding the structure, distribution, and characteristics of the BirdCLEF 2026 dataset will inform modeling decisions — particularly around the new labeled soundscape data, multi-taxa handling, class imbalance, and the expanded label space.

## Background

BirdCLEF 2026 introduces two major changes from prior years:

1. **Labeled train soundscapes** — for the first time, the competition provides `train_soundscapes_labels.csv` with per-5s-window multi-species annotations for 10,658 soundscape recordings from the Pantanal, Brazil. Previous BirdCLEF editions only provided unlabeled soundscapes for inference; labeled soundscapes open the door to supervised SED training and domain adaptation.

2. **Multi-taxa expansion** — the dataset now covers 5 taxonomic classes (162 Aves, 35 Amphibia, 28 Insecta, 8 Mammalia, 1 Reptilia) totaling 234 taxonomy entries, up from birds-only in prior years. Labels are a mix of eBird-style codes (187 species) and iNat numeric taxon IDs (47 species), with one species (47158) split into 25 song-type sub-labels.

The training data includes ~35K focal recordings across 206 species directories (28 taxonomy entries lack training audio). Before building models, we need a clear picture of the data landscape.

### Key Questions

- Which 28 taxonomy species have no training audio? Are they soundscape-only or test-only?
- How dense are soundscape labels — how many species co-occur per 5s window?
- How do soundscape species overlap with focal training species?
- What's the class imbalance across species and across taxa?
- How do iNat vs XenoCanto recordings differ in quality and coverage?
- What's the structure of the 47158 song-type splits?

## Method

### Approach

Systematic EDA covering metadata, label distributions, soundscape characteristics, and audio properties. Output a Jupyter notebook with visualizations and summary statistics in `artifacts/`.

### Setup

| Component | Details |
|-----------|---------|
| Data | `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/` |
| Files | train.csv (35,549 rows), taxonomy.csv (234 entries), train_soundscapes_labels.csv (1,478 label rows), train_soundscapes/ (10,658 files), train_audio/ (206 species dirs), sample_submission.csv (234 target columns), recording_location.txt (Pantanal, Brazil) |
| Compute | CPU only (login node or 1-core interactive) |
| Dependencies | pandas, matplotlib, seaborn, librosa (audio stats), jupyter |
| Code | `user/anthony/experiments/200-eda/` |
| Outputs | `artifacts/` — plots (PNG), summary notebook |

### Procedure

**Metadata (fast, no audio loading):**

1. **Schema overview** — column types, missing values, unique counts for all CSVs
2. **Taxonomy breakdown** — species per class, label format (eBird vs iNat numeric vs song-type splits), taxonomy-vs-training coverage gap (28 missing species)
3. **Class distribution** — recordings per species, imbalance ratio, long-tail analysis, per-taxon breakdown
4. **Collection source** — iNat vs XenoCanto split, rating distribution by source and taxon
5. **Geographic analysis** — lat/lon coverage, missing coordinates, species by region
6. **Secondary labels** — co-occurrence frequency, multi-label density in focal recordings
7. **Recording type** — distribution of `type` column (call, song, etc.) by taxon

**Soundscape analysis (new for 2026):**

8. **Soundscape label structure** — label density per 5s window, unique species per window distribution, temporal label stability across consecutive windows
9. **Soundscape species coverage** — which species appear in soundscapes vs focal-only? How many soundscape files per species?
10. **Focal-soundscape domain gap** — compare species distributions, any soundscape-only species?

**Audio properties (requires file sampling):**

11. **Duration distribution** — sample files for duration stats (mean, median, std), per-taxon differences
12. **Format check** — sample rate, channels, codec consistency across sources

### Variables

- **Independent**: data subset (focal vs soundscape, per-taxon, per-source)
- **Dependent**: distribution statistics, visualizations
- **Controlled**: N/A (observational)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Class imbalance ratio | max/min recordings per species |
| Soundscape label density | avg unique species per 5s window |
| Coverage gap | species in taxonomy but missing from train_audio |
| Soundscape overlap | fraction of soundscape species also in focal training |
| Duration stats | mean, median, std of audio lengths |

### Baseline

No baseline — this is exploratory/descriptive.

### Success Criteria

- **Confirm if**: All 12 procedure steps produce summary statistics and/or visualizations
- **Reject if**: N/A (EDA is always informative)

## Limitations

- Audio property analysis requires sampling (not feasible to load all 35K+ files on login node)
- Geographic analysis limited to provided lat/lon (may have missing values)
- No access to hidden test soundscapes for distribution comparison
- Soundscape label quality is unverified — labels may have noise or systematic biases
