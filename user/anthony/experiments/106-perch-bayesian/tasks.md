# Tasks

## 1. Setup
- [x] 1.1 Create package skeleton (pyproject.toml, src/)
- [x] 1.2 Copy perch_onnx.py and audio utilities from exp 105
- [x] 1.3 Write extraction script (polars-based, all three sources)
- [x] 1.4 Write sbatch scripts for extraction jobs
- [x] 1.5 Download Perch v2 label mapping from Kaggle model
- [x] 1.6 Copy species mapping from shared artifacts

## 2. Extraction
- [x] 2.1 Run labeled soundscape extraction (739 rows, 43 MB)
- [x] 2.2 Run unlabeled soundscape extraction (127,104 rows, 7.2 GB, 16 shards)
- [x] 2.3 Run focal extraction (265,924 rows, 15 GB, 16 shards)
- [x] 2.4 Verify output parquets (row counts, schema)

## 3. Phase 1 — Static analysis
- [x] 3.1 Zero-shot baseline: direct Perch logit readout for 203 mapped species
- [x] 3.2 Distribution shift diagnostics (focal vs soundscape, KS tests)
- [x] 3.3 Prototype classifier for insect sonotypes (nearest centroid in embedding space)
- [x] 3.4 Feature selection (per-logit AUROC, exploratory)
- [x] 3.5 Logit covariance + embedding covariance from unlabeled soundscapes

## 4. Phase 2 — Models + temporal
- [x] 4.1 Empirical Bayes ridge regression (6 configs, LOSO CV) — best: 0.749 AUROC
- [x] 4.2 Sparse logistic regression (per-species L1) — 0.298 AUROC (not competitive)
- [x] 4.3 Temporal pooling (max/mean/softmax on ridge OOF) — 0.698 clip AUROC
- [x] 4.4 Soundscape profiles (per-site + diel signatures from unlabeled data)
- [x] 4.5 Focal species profiles (per-species activation signatures)
- [ ] 4.6 DMD on per-site time series — FAILED (bug: `pca` → `global_pca` line 162)
- [ ] 4.7 Compare: embeddings vs logits vs PCA-reduced vs concatenated

## 5. Phase 3 — Full pipeline
- [ ] 5.1 Fix DMD bug and rerun
- [ ] 5.2 UKF temporal smoothing (DMD-informed dynamics + ridge observation model)
- [ ] 5.3 Classification-occupancy model (calibrate FP/FN rates)
- [ ] 5.4 GP classification on embeddings (if ridge saturates)

## 6. Documentation
- [x] 6.1 Write analysis ideas doc with references
- [x] 6.2 Document Phase 1 results in results.md
- [x] 6.3 Document Phase 2 results in results.md
