# Results: Teacher Labels for BirdCLEF 2025

**Date Completed**: 2026-02-16
**Author**: Anthony

## Summary

All three teacher models (BamBird, BirdNET, Perch) ran on the full BirdCLEF 2025 dataset (28,564 focal recordings + 9,726 soundscapes) via CPU-only SLURM array jobs. BirdNET and Perch completed successfully with usable outputs. **BamBird detected zero events** — its ROI extraction pipeline failed silently on all files.

**Verdict**: partial — BirdNET and Perch labels ready for fusion; BamBird needs investigation

## Job Completion

| Model | Focal Parts | Soundscape Parts | Status |
|-------|-------------|-----------------|--------|
| BamBird | 50/50 | 20/20 | Complete (0 ROIs — broken) |
| BirdNET | 50/50 | 20/20 | Complete |
| Perch | 50/50 | 20/20 | Complete |

## Statistics (from `teacher-stats`)

### BamBird: Zero Event Detection

| Source | Files | Total ROIs | Files with ROIs |
|--------|-------|-----------|----------------|
| Focal | 28,564 | 0 | 0 |
| Soundscapes | 9,726 | 0 | 0 |

BamBird's ROI extraction found nothing in any file. This was also visible in the benchmark (0 ROIs for all 5 test files). The scikit-maad spectrogram → ROI pipeline may need parameter tuning, or the audio format/sampling rate is incompatible with BamBird's defaults. Total audio: 280.5h focal + 162.1h soundscapes = **442.6 hours**.

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

Focal recordings have much higher peak confidence than soundscapes (expected — focal recordings target a single species). Top species by mean logit (focal): 3189, 5336, 5999, 5532, 1796 (numeric IDs — see note below).

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

Perch shows higher absolute confidence scores than BirdNET on focal recordings (P95: 8.23 vs 5.88).

### Model Agreement

| Source | BirdNET Species | Perch Species | Common Species | BirdNET Clips | Perch Clips |
|--------|----------------|--------------|---------------|--------------|-------------|
| Focal | 6,522 | 10,932 | 6,522 | 939,116 | 187,839 |
| Soundscapes | 6,522 | 10,932 | 6,522 | 564,108 | 116,712 |

All 6,522 BirdNET species exist in Perch's 10,932 species (BirdNET is a subset of Perch). BirdNET produces ~5x more clips than Perch due to finer temporal resolution (3s/1s vs 5s/5s).

## Timing

### Benchmark (5-file sample, pre-run)

| Model | Per-file (s) | Extrapolated wall time/partition | Total (50 partitions) |
|-------|-------------|--------------------------------|----------------------|
| BamBird | 0.31 | 3.9 min | 0.07 h |
| BirdNET | 10.4 | 133 min | 2.2 h |
| Perch | 27.7 | 354 min | 5.9 h |

### Actual (from first 3 partition timing files)

| Model | Source | Files/part | Time/part | Per-file (s) |
|-------|--------|-----------|-----------|-------------|
| BamBird | focal | ~572 | ~4 min | 0.3 |
| BirdNET | focal | 572 | ~32 min | 3.3 |
| BirdNET | soundscape | 487 | ~46 min | 5.7 |
| Perch | focal | 572 | ~91 min | 9.5 |
| Perch | soundscape | 487 | ~137 min | 16.8 |

Actual per-file times are 2-3x faster than the 5-file benchmark, likely due to model loading overhead amortizing over more files. Perch on CPU via TFLite was feasible — ~10-17s per file.

## Storage

| Model | Component | Focal | Soundscape | Total |
|-------|-----------|-------|------------|-------|
| BamBird | events | 804 KB | 244 KB | **1.1 MB** |
| BirdNET | embed (1024-dim) | 4.6 GB | 2.7 GB | 7.3 GB |
| BirdNET | predict (6522 species) | 36.1 GB | 21.7 GB | 57.8 GB |
| Perch | embed (1280-dim) | 1.4 GB | 854 MB | 2.2 GB |
| Perch | predict (10932 species) | 15.3 GB | 9.6 GB | 24.9 GB |
| **Total** | | | | **~90 GB** |

Storage is dominated by predict parquets (82.7 GB / 92.2 GB). Could reduce by:
- Filtering to BirdCLEF 2025 species only (~206 species vs 6522/10932)
- Storing only top-K predictions per clip
- Using larger clip step for BirdNET (current: 1s step → lots of overlap)

## Output Schemas

### BamBird
- Format: `[file, events_json, num_rois, duration_s]`
- `events_json` contains variable-length ROI lists as JSON strings
- **All empty** (0 ROIs everywhere)

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

## Observations

### Key Findings

1. **BamBird is non-functional** — Zero ROIs across 38,290 files (442.6 hours of audio). The scikit-maad ROI extraction pipeline silently returns empty results. Needs parameter tuning or debugging.

2. **Species IDs are numeric, not names** — Both BirdNET and Perch use internal integer indices. A mapping from indices to scientific/common names is needed for interpretability.

3. **Logits, not probabilities** — Confidence values are raw model outputs (logits). For downstream use, apply sigmoid (binary) or softmax (multi-class) to convert to probabilities.

4. **Focal > Soundscapes in confidence** — BirdNET max-conf P95 is 5.88 for focal vs 0.08 for soundscapes. Perch: 8.23 vs 1.63. Focal recordings (targeted single-species) produce much clearer predictions than soundscapes (ambient multi-species).

5. **BirdNET provides 5x temporal resolution** — 939K clips at 3s/1s step vs 188K at 5s/5s step for focal. More data points for label fusion but also more storage.

6. **All BirdNET species exist in Perch** — 6,522 common species. Perch has 4,410 additional species. This is good for fusion — predictions can be aligned on the common species set.

## Artifacts

| Type | Location |
|------|----------|
| Stats JSON | `~/scratch/birdclef-2026/outputs/exp010/stats.json` |
| Activity JSON | `~/scratch/birdclef-2026/outputs/exp010/activity.json` |
| Stats Plot | `~/scratch/birdclef-2026/outputs/exp010/teacher_stats.png` |
| Benchmark | `~/scratch/birdclef-2026/outputs/exp010/benchmark.json` |
| Perch TFLite | `~/scratch/birdclef-2026/outputs/exp010/perch.tflite` |
| BamBird labels | `~/scratch/birdclef-2026/outputs/exp010/labels/bambird/` |
| BirdNET labels | `~/scratch/birdclef-2026/outputs/exp010/labels/birdnet/` |
| Perch labels | `~/scratch/birdclef-2026/outputs/exp010/labels/perch/` |
| SLURM logs | `user/anthony/experiments/010-teacher-labels/logs/` |

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

**Max-prob percentiles (focal):**
- BirdNET: P25=0.096, P50=0.399, P75=0.907, P90=0.990, P99=1.000
- Perch: P25=0.238, P50=0.833, P75=0.990, P90=0.999, P99=1.000

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

**Max-prob percentiles (soundscapes):**
- BirdNET: P25=0.046, P50=0.091, P75=0.185, P90=0.353, P99=0.884
- Perch: P25=0.035, P50=0.072, P75=0.192, P90=0.553, P99=0.993

### Activity Observations

1. **Focal recordings are ~45-64% active** at the 0.50 threshold. Perch is more confident (median prob 0.83 vs BirdNET 0.40) — it makes sharper predictions.

2. **Soundscapes are mostly quiet**: only 5-11% of segments exceed 0.50 confidence. Median max-prob is just 0.07-0.09. Most 3-5s windows in ambient recordings are background noise, wind, or rain.

3. **Perch is more decisive than BirdNET**: at the 0.99 threshold, Perch still classifies 24.6% of focal clips as active (vs BirdNET 10.1%). On soundscapes, Perch finds 1.2% at 0.99 vs BirdNET 0.1%.

4. **For downstream label fusion**, a threshold around 0.50 is reasonable for focal, but soundscapes need a lower bar (~0.10-0.25) to capture meaningful activity, or per-species calibration.

## Success Criteria Evaluation

1. **All partitions complete without error** — Yes, all 240 partition jobs finished.
2. **Schemas consistent and compatible with downstream fusion** — Yes for BirdNET/Perch. BamBird outputs are empty.
3. **BirdNET and Perch top-species predictions correlate** — Species overlap confirmed (6,522 common). Detailed per-clip agreement not yet computed (requires aligning temporal windows).
4. **Total storage < 20 GB** — No, actual is ~90 GB (predict parquets are large).

## Next Steps

- [ ] Investigate BamBird zero-ROI issue (parameter tuning, audio format check)
- [ ] Map numeric species IDs to scientific names for BirdNET and Perch
- [ ] Filter predict parquets to BirdCLEF 2025 species (~206) to reduce storage from 90 GB to ~3 GB
- [ ] Compute per-clip BirdNET vs Perch agreement on aligned temporal windows
- [x] ~~Convert logits to probabilities and set confidence thresholds~~ (done — activity analysis with sigmoid thresholds)
- [ ] Integrate with label fusion pipeline (`docs/concepts/weak-supervision-label-fusion.md`)

**Leads to**: Label fusion via FlyingSquid (triplet method), SED integration in exp 009 finetuning pipeline
