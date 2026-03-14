# Proposal: BirdCLEF 2026 Exploratory Data Analysis

**Status**: proposed
**Created**: 2026-03-14
**Author**: Anthony

## Hypothesis

This EDA will produce actionable artifacts — a label mapping, class weights, and data summary — that all downstream experiments consume. Without these, modeling decisions (split strategy, loss weighting, frequency config, soundscape inclusion) are guesswork.

## Background

BirdCLEF 2026 introduces two major changes from prior years:

1. **Labeled train soundscapes (new)** — for the first time, the competition provides `train_soundscapes_labels.csv` with per-5s-window multi-species annotations. However, only **66 of 10,658 soundscape files** (0.6%) have labels (1,478 label rows). The remaining 99.4% are unlabeled. This severely limits supervised SED on soundscapes and suggests semi-supervised or pseudo-labeling approaches. Previous BirdCLEF editions only provided unlabeled soundscapes for inference.

2. **Multi-taxa expansion** — the dataset now covers 5 taxonomic classes (162 Aves, 35 Amphibia, 28 Insecta, 8 Mammalia, 1 Reptilia) totaling 234 taxonomy entries, up from birds-only in prior years. Labels are a mix of eBird-style codes (187 species) and iNat numeric taxon IDs (47 species), with one species (47158) split into 25 song-type sub-labels in the submission.

The training data includes ~35K focal recordings across 206 species directories. Of the 28 taxonomy entries without a `train_audio/` directory, 25 are the `47158sonXX` song-type splits (which share the `47158/` directory) and 3 are genuinely missing species (25073, 517063, 1491113). The submission target space is 234 columns.

### Key Questions

**Blocking (required before any modeling):**
- What is the exact label mapping from `primary_label` → submission column index? (eBird codes, numeric IDs, and 47158 song-type splits all coexist)
- How are the 25 `47158sonXX` song-type submission columns related to the single `47158/` training directory? Is there metadata to assign recordings to song types?
- Which 3 non-songtype species have zero training audio? Do they appear in soundscape labels?

**Soundscape (new data source):**
- Only 66/10,658 soundscapes are labeled — what is the species coverage? (75 unique species in labels)
- How dense are labels — how many species co-occur per 5s window?
- Are unlabeled windows in labeled files true negatives, or just unlabeled?
- How do soundscape species overlap with focal training species?

**Modeling decisions:**
- What's the class imbalance across species and across taxa?
- How do iNat vs XenoCanto recordings differ in quality and coverage?
- What are the spectral characteristics by taxon? (insects vs birds vs amphibians need different frequency ranges)
- How much geographic overlap is there between focal training data and the Pantanal test site?
- How many BirdCLEF 2025 species carry over to 2026? (transfer learning viability)

## Method

### Approach

Luigi pipeline producing derived parquet datasets, plus a label mapping JSON. Each task reads raw competition data and outputs to `artifacts/`. Priority-ordered: label mapping and soundscape analysis first, audio properties last.

### Setup

| Component | Details |
|-----------|---------|
| Data | `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/` |
| Files | train.csv (35,549 rows), taxonomy.csv (234 entries), train_soundscapes_labels.csv (1,478 label rows across 66 files), train_soundscapes/ (10,658 files), train_audio/ (206 species dirs), sample_submission.csv (234 target columns), recording_location.txt (Pantanal, Brazil) |
| Compute | CPU sbatch job (8 cores, 32GB, ~15 min expected) |
| Dependencies | polars, luigi, soundfile, matplotlib, seaborn |
| Code | `user/anthony/experiments/100-eda/` |
| Outputs | `artifacts/` — derived parquets, label_mapping.json |

### Procedure

**Priority 1 — Label infrastructure (blocks all modeling):**

1. **Label mapping** — build mapping from every `primary_label` in train.csv, taxonomy.csv, and soundscape labels to submission column index. Handle eBird codes, numeric IDs, and 47158 song-type splits. Output: `label_mapping.json`
2. **Song-type investigation** — analyze species 47158: what recordings exist in `train_audio/47158/`, is there metadata to assign them to the 25 `sonXX` sub-types?
3. **Schema overview** — column types, missing values, unique counts for all CSVs

**Priority 2 — Soundscape analysis (new data source):**

4. **Soundscape label structure** — label density per 5s window, species per window distribution, temporal label stability across consecutive windows within files
5. **Soundscape species coverage** — which 75 species appear in soundscape labels? How many are focal-only, soundscape-only, or both?
6. **Negative label analysis** — are there windows with no labels in labeled files? Can unlabeled windows be used as hard negatives?

**Priority 3 — Training data characterization:**

7. **Taxonomy breakdown** — species per class, label format distribution, the 3 genuinely missing species
8. **Class distribution** — recordings per species, imbalance ratio, long-tail analysis, per-taxon breakdown
9. **Collection source** — iNat vs XenoCanto split, rating distribution by source and taxon (note: iNat ratings are all 0.0)
10. **Secondary labels** — co-occurrence frequency, multi-label density in focal recordings

**Priority 4 — Domain gap analysis:**

11. **Geographic analysis** — lat/lon coverage of focal data vs Pantanal test site coordinates, missing coordinate rate
12. **Species overlap with BirdCLEF 2025** — how many of the 234 species appeared in the 2025 competition? Determines transfer learning viability.
13. **Test set structure** — extract site IDs, dates, and time offsets from sample_submission row IDs

**Priority 5 — Audio properties (requires file I/O):**

14. **Duration distribution** — measure all focal audio durations, per-taxon differences
15. **Soundscape file inventory** — total duration of labeled vs unlabeled soundscapes

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Label mapping completeness | all 234 submission columns mapped |
| Class imbalance ratio | max/min recordings per species |
| Soundscape label density | avg unique species per 5s window |
| Soundscape coverage | fraction of 234 species appearing in soundscape labels |
| Geographic overlap | fraction of focal recordings within Pantanal region |

### Baseline

No baseline — this is exploratory/descriptive.

### Success Criteria

- **Confirm if**: label_mapping.json covers all 234 submission columns, and all priority 1-3 steps produce derived parquets
- **Reject if**: N/A (EDA is always informative)

## Deliverables

1. `artifacts/label_mapping.json` — primary_label → submission column index (all 234)
2. `artifacts/train_metadata.parquet` — enriched train.csv with taxonomy join
3. `artifacts/taxonomy_coverage.parquet` — per-species coverage analysis
4. `artifacts/class_distribution.parquet` — per-species × collection statistics
5. `artifacts/soundscape_labels.parquet` — exploded soundscape labels
6. `artifacts/soundscape_window_stats.parquet` — per-window species density
7. `artifacts/soundscape_coverage.parquet` — focal vs soundscape species overlap
8. `artifacts/audio_durations.parquet` — duration of all focal recordings

## Limitations

- Only 0.6% of soundscapes are labeled — soundscape analysis is limited to 66 files
- Geographic analysis limited to provided lat/lon (may have missing values)
- No access to hidden test soundscapes for distribution comparison
- Song-type assignment for 47158 may be impossible without external metadata
- Spectral analysis by taxon is deferred to a follow-up (requires audio loading + visualization)
