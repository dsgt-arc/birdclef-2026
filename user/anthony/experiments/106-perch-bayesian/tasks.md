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
- [ ] 3.1 Zero-shot baseline: direct Perch logit readout for 203 mapped species
- [ ] 3.2 Empirical Bayes ridge on embeddings (unlabeled covariance as prior)
- [ ] 3.3 Distribution shift diagnostics (focal vs soundscape, UMAP, KS tests)
- [ ] 3.4 Prototype classifier for insect sonotypes (nearest centroid in embedding space)
- [ ] 3.5 Compare: embeddings vs logits vs PCA-reduced logits vs concatenated

## 4. Phase 2 — Temporal analysis
- [ ] 4.1 Simple temporal baselines (max-pool, mean-pool ridge predictions across 12 windows)
- [ ] 4.2 HMM diel regimes on unlabeled data (per-site Gaussian HMM)
- [ ] 4.3 DMD on per-site embedding time series (discover dynamical modes)
- [ ] 4.4 Site/temporal activation patterns (per-site profiles, diel analysis)

## 5. Phase 3 — Full pipeline
- [ ] 5.1 UKF temporal smoothing (DMD-informed dynamics + ridge observation model)
- [ ] 5.2 Classification-occupancy model (calibrate FP/FN rates)
- [ ] 5.3 GP classification on embeddings (if ridge saturates)
- [ ] 5.4 DMD mode analysis and transition matrix clustering

## 6. Documentation
- [x] 6.1 Write analysis ideas doc with references
- [ ] 6.2 Document results in results.md
