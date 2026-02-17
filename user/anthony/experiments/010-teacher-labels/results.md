# Results: Teacher Labels for BirdCLEF 2025

**Date Completed**: 2026-02-17
**Author**: Anthony

## Summary

All three teacher models (BamBird, BirdNET, Perch) ran on the full BirdCLEF 2025 dataset (28,564 focal recordings + 9,726 soundscapes) via CPU-only SLURM array jobs. All pipelines completed successfully.

**Verdict**: complete — all teacher labels ready for fusion

## Output Directory

```
~/scratch/birdclef-2026/outputs/exp010/          (~88 GB total)
├── activity.json                                 (4.7 KB)
├── benchmark.json                                (2.2 KB)
├── perch.tflite                                  (25 MB)
├── stats.json                                    (8.7 KB)
├── teacher_stats.png                             (144 KB)
└── labels/
    ├── bambird/                                  (2.1 GB)
    │   ├── features/                             (2.1 GB)
    │   │   ├── part_0000/                        (28,564 parquets across 50 shards)
    │   │   │   ├── species__stem.parquet          ~45-53 KB each, 571-572 per shard
    │   │   │   └── ...
    │   │   └── part_0049/
    │   ├── train_audio/                          (19 MB)
    │   │   ├── bambird_labels.parquet             (6.5 MB — clustered output)
    │   │   └── part_NNNN.parquet                  (50 files, ~230-300 KB each — old extract)
    │   └── train_soundscapes/                    (5.7 MB)
    │       └── part_NNNN.parquet                  (20 files — old extract)
    ├── birdnet/                                  (61 GB)
    │   ├── train_audio/parts/                    (38 GB)
    │   │   ├── predict/                           (34 GB — 50 parquets, ~690 MB each)
    │   │   ├── embed/                             (4.3 GB — 50 parquets)
    │   │   └── timing/                            (204 KB — 50 parquets)
    │   └── train_soundscapes/parts/              (23 GB)
    │       ├── predict/                           (21 GB — 20 parquets)
    │       ├── embed/                             (2.6 GB — 20 parquets)
    │       └── timing/                            (84 KB — 20 parquets)
    └── perch/                                    (26 GB)
        ├── train_audio/parts/                    (16 GB)
        │   ├── predict/                           (15 GB — 50 parquets)
        │   ├── embed/                             (1.3 GB — 50 parquets)
        │   └── timing/                            (204 KB — 50 parquets)
        └── train_soundscapes/parts/              (9.8 GB)
            ├── predict/                           (9.0 GB — 20 parquets)
            ├── embed/                             (815 MB — 20 parquets)
            └── timing/                            (84 KB — 20 parquets)
```

## Job Completion

| Model | Focal Parts | Soundscape Parts | Status |
|-------|-------------|-----------------|--------|
| BamBird extract | 50/50 | 20/20 | Complete |
| BamBird cluster | 1/1 | N/A | Complete |
| BirdNET | 50/50 | 20/20 | Complete |
| Perch | 50/50 | 20/20 | Complete |

## Statistics

### BamBird: ROI Extraction + Clustering

| Metric | Focal |
|--------|-------|
| Files processed | 28,564 |
| Files with ROIs | 21,814 (76%) |
| Total ROIs | 660,081 |
| ROIs per file (mean) | 30.3 |
| ROIs per file (median) | 5.0 |
| Event duration (mean) | 0.80s |
| Event duration (median) | 0.44s |
| Species | 201 |

**Clustering results (KEEP=BIGGEST DBSCAN):**

| Metric | Value |
|--------|-------|
| Signal ROIs | 81,537 (12.4%) |
| Noise ROIs | 578,544 (87.6%) |
| Files with 0 signal | 12,149 (55.7%) |
| Species with 0% signal | 56/201 (34 birds + 22 non-Aves) |

**Clustering quality concerns:**
- 34 bird species have 0% signal — all ROIs labeled as DBSCAN outliers (cluster -1)
- Root cause: auto-tuned `min_points` too high for species with uniform acoustics; DBSCAN can't form dense clusters, labels everything as noise
- `KEEP: BIGGEST` discards valid secondary call types (only largest cluster = signal)
- Signal ratio distribution: P25=0.00, P50=0.14, P75=0.29
- **Clustering output is not consumed by any downstream pipeline** — the temporal event boundaries from extract stage are the useful output

**Note**: BamBird only runs on focal recordings (species label needed for per-species clustering). Soundscape extract exists from an older run but clustering is not applicable without species labels.

### BirdNET: Species Classification

| Source | Clips | Species | Total Time |
|--------|-------|---------|-----------|
| Focal (3s/1s step) | 939,116 | 6,522 | 26.6h |
| Soundscapes (3s/1s step) | 564,108 | 6,522 | 15.9h |

**Max confidence per clip (raw logits, pre-sigmoid):**

| Source | Mean | Median | P95 |
|--------|------|--------|-----|
| Focal | 0.20 | -0.41 | 5.88 |
| Soundscapes | -2.16 | -2.30 | 0.08 |

### Perch: Species Classification + Embeddings

| Source | Clips | Species | Total Time |
|--------|-------|---------|-----------|
| Focal (5s step) | 187,839 | 10,932 | 73.9h |
| Soundscapes (5s step) | 116,712 | 10,932 | 45.8h |

**Max confidence per clip (raw logits):**

| Source | Mean | Median | P95 |
|--------|------|--------|-----|
| Focal | 1.88 | 1.61 | 8.23 |
| Soundscapes | -2.11 | -2.55 | 1.63 |

### Model Agreement

| Source | BirdNET Species | Perch Species | Common Species | BirdNET Clips | Perch Clips |
|--------|----------------|--------------|---------------|--------------|-------------|
| Focal | 6,522 | 10,932 | 6,522 | 939,116 | 187,839 |
| Soundscapes | 6,522 | 10,932 | 6,522 | 564,108 | 116,712 |

All 6,522 BirdNET species exist in Perch's 10,932 species (BirdNET is a subset). BirdNET produces ~5x more clips due to finer temporal resolution (3s/1s vs 5s/5s).

## Storage

| Model | Component | Focal | Soundscape | Total |
|-------|-----------|-------|------------|-------|
| BamBird | features (96-dim wavelet) | 2.1 GB | — | 2.1 GB |
| BamBird | cluster labels | 6.5 MB | — | 6.5 MB |
| BirdNET | embed (1024-dim) | 4.3 GB | 2.6 GB | 6.9 GB |
| BirdNET | predict (6522 species) | 34 GB | 21 GB | 55 GB |
| Perch | embed (1280-dim) | 1.3 GB | 815 MB | 2.1 GB |
| Perch | predict (10932 species) | 15 GB | 9.0 GB | 24 GB |
| **Total** | | | | **~88 GB** |

Storage is dominated by predict parquets (79 GB / 88 GB). Could reduce by:
- Filtering to BirdCLEF 2025 species only (~206 species vs 6522/10932)
- Storing only top-K predictions per clip
- Using larger clip step for BirdNET (current: 1s step with lots of overlap)

## Output Schemas

### BamBird Features (per-file parquets)
- One parquet per audio file in `features/part_NNNN/species__stem.parquet`
- Columns: 96 wavelet shape features, ROI time/freq bounds (`min_t`, `max_t`, `min_f`, `max_f`), metadata (`filename`, `species`, `categories`, `abs_min_t`)
- Empty parquets for files with no detected ROIs

### BamBird Cluster Output (`bambird_labels.parquet`)
- Schema: `[file, species, detected_events, event_cluster, num_signal_rois, num_noise_rois]`
- `detected_events`: JSON array of `[min_t, max_t]` pairs
- `event_cluster`: JSON array of cluster IDs (-1 = noise, 0+ = clusters)
- 21,814 rows (one per file with ROIs)

### BirdNET (via bioacoustics_model_zoo)
- Embed: `(939K, 1024)` focal + `(564K, 1024)` soundscape
- Predict: `(939K, 6522)` focal + `(564K, 6522)` soundscape
- Values are **raw logits** (not probabilities) — need sigmoid for probabilities
- Species columns are **numeric IDs** (e.g., "3189"), not species names

### Perch (TFLite)
- Embed: `(188K, 1280)` focal + `(117K, 1280)` soundscape
- Predict: `(188K, 10932)` focal + `(117K, 10932)` soundscape
- Values are **raw logits** — need sigmoid for probabilities
- Species columns are **numeric IDs** (e.g., "9321")
- Compiled TFLite model: `perch.tflite` (24.3 MB) with SELECT_TF_OPS

## Timing

### Benchmark (5-file sample, pre-run)

| Model | Per-file (s) | Extrapolated wall time/partition | Total (50 partitions) |
|-------|-------------|--------------------------------|----------------------|
| BamBird | 0.31 | 3.9 min | 0.07 h |
| BirdNET | 10.4 | 133 min | 2.2 h |
| Perch | 27.7 | 354 min | 5.9 h |

### Actual (from partition timing files)

| Model | Source | Files/part | Time/part | Per-file (s) |
|-------|--------|-----------|-----------|-------------|
| BamBird | focal | ~572 | ~105 min | 11.0 |
| BirdNET | focal | 572 | ~32 min | 3.3 |
| BirdNET | soundscape | 487 | ~46 min | 5.7 |
| Perch | focal | 572 | ~91 min | 9.5 |
| Perch | soundscape | 487 | ~137 min | 16.8 |

BamBird extract (with bambird library) is slower than the original maad-only approach but produces valid results. BirdNET and Perch actual times are 2-3x faster than benchmark due to model loading overhead amortizing.

## Activity Analysis

Segments classified as "active" when the max species probability (sigmoid of logit) exceeds a given threshold.

### Focal Recordings

| Threshold | BirdNET (939K clips) | % | Perch (188K clips) | % |
|-----------|---------------------|---|-------------------|---|
| 0.10 | 697,744 | 74.3% | 159,422 | 84.9% |
| 0.25 | 547,156 | 58.3% | 139,757 | 74.4% |
| **0.50** | **427,716** | **45.5%** | **120,981** | **64.4%** |
| 0.75 | 327,642 | 34.9% | 102,391 | 54.5% |
| 0.90 | 240,320 | 25.6% | 83,984 | 44.7% |
| 0.95 | 188,037 | 20.0% | 71,695 | 38.2% |
| 0.99 | 94,732 | 10.1% | 46,212 | 24.6% |

### Soundscapes

| Threshold | BirdNET (564K clips) | % | Perch (117K clips) | % |
|-----------|---------------------|---|-------------------|---|
| 0.10 | 262,132 | 46.5% | 46,912 | 40.2% |
| 0.25 | 95,204 | 16.9% | 23,775 | 20.4% |
| **0.50** | **30,393** | **5.4%** | **12,992** | **11.1%** |
| 0.75 | 11,569 | 2.0% | 7,616 | 6.5% |
| 0.90 | 4,969 | 0.9% | 4,440 | 3.8% |
| 0.95 | 2,821 | 0.5% | 3,144 | 2.7% |
| 0.99 | 734 | 0.1% | 1,429 | 1.2% |

## Observations

1. **BamBird provides temporal boundaries, not reliable signal/noise labels** — The DBSCAN clustering fails for 16% of bird species and assigns only 12.4% of ROIs as signal. The valuable output is the variable-length event boundaries from the extract stage, not the cluster labels.

2. **BamBird as weak labeling function** — Despite unreliable signal/noise classification, BamBird's acoustic-feature-based labels could serve as a weak labeling function in FlyingSquid alongside BirdNET/Perch. The three teachers have independent error modes (unsupervised wavelet features vs two different neural classifiers), which is ideal for weak supervision.

3. **Species IDs are numeric, not names** — Both BirdNET and Perch use internal integer indices. A mapping from indices to scientific/common names is needed for interpretability.

4. **Logits, not probabilities** — Confidence values are raw model outputs. Apply sigmoid for probabilities.

5. **Focal >> Soundscapes in confidence** — BirdNET max-conf P95 is 5.88 for focal vs 0.08 for soundscapes. Soundscapes are mostly quiet (only 5-11% of clips exceed 0.50 confidence).

6. **Perch is more decisive than BirdNET** — At 0.99 threshold, Perch still classifies 24.6% of focal clips as active (vs BirdNET 10.1%).

7. **BirdNET provides 5x temporal resolution** — 939K clips at 3s/1s step vs 188K at 5s/5s step. More data points for label fusion but also more storage.

8. **All BirdNET species exist in Perch** — 6,522 common species. Predictions can be aligned on the common species set.

## Next Steps

- [ ] Map numeric species IDs to scientific names for BirdNET and Perch
- [ ] Filter predict parquets to BirdCLEF 2025 species (~206) to reduce storage from 88 GB to ~3 GB
- [ ] Compute per-clip BirdNET vs Perch agreement on aligned temporal windows
- [ ] Integrate with label fusion pipeline (FlyingSquid — BamBird + BirdNET + Perch as 3 LFs)
- [ ] Run BamBird extract on soundscapes (ROI boundaries useful even without clustering)
- [x] ~~BamBird ROI extraction~~ (fixed — rewrote with bambird library API)
- [x] ~~Convert logits to probabilities and set confidence thresholds~~ (activity analysis done)

**Leads to**: Label fusion via FlyingSquid (triplet method), SED integration in exp 011 prompted SED pipeline
