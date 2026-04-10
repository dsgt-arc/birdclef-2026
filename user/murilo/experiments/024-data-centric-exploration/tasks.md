# Experiment 024: Data-Centric Exploration — Tasks

## Goal

Understand model coverage gaps and Perch/SED complementarity to inform the pseudo-labeling
teacher strategy for exp 025.

Status: **Phase A complete. Phase B complete (both train_soundscapes and train_audio verified). Phase C unblocked.**

**Branch**: `mg/024-data-centric-exploration` — PR #26

---

## Phase A: Max-Across-Folds Analysis ✅

- [x] A.1 Compute max(fold0..fold3) from existing 4-fold parquets
- [x] A.2 Compare coverage: avg-folds vs max-folds vs single-fold for 159 unseen species
  - Max-folds: 109/159 at >0.5 (+29 vs single, +41 vs avg)
  - 34 species newly recovered (single=0, max>0)
  - Only 50 species remain at zero (vs 79 single, 91 avg)
  - 34 truly hard species at >0.3 — need Perch
- [x] A.3 Update inference EDA doc (section 7 rewritten) and inference_eda.py script

## Phase B: Perch v2 TF Inference 🔄

- [x] B.1 Write standalone `inference/perch_inference.py` (TF SavedModel, saves embeddings + logits)
- [x] B.2 Write SLURM batch scripts (`inference/sbatch/perch_inference.sh` + `run_perch.sh`)
- [x] B.3 Run inference → save embeddings + raw logits to `shared/inference/birdclef-2026/perch-v2-tf/`
  - ✅ Test: 10 files (TF version fix), 200 files (throughput benchmark: 10.7 chunks/s)
  - ✅ Done: train_soundscapes (10,658 files) — 127,896 rows (12 chunks/file)
  - ✅ Done: train_audio (35,549 files) — 265,924 rows (variable chunks/file)
  - Submit: `cd user/murilo/inference && bash sbatch/run_perch.sh --source <source> --overwrite`
  - Config: cpu-medium, 24 CPUs, 64GB RAM, 24h walltime, TF 2.21 CPU
  - Note: initial train_audio job OOM'd at 70% due to read-concat-write flush pattern.
    Fixed `perch_inference.py` to use chunked parquet writes + streaming merge.
    Added resume support (skips already-processed files). Resumed and completed.
- [x] B.4 Verify output shapes and sample predictions (train_soundscapes)
  - embeddings.parquet: 127,896 rows × 1,541 cols (5 metadata + 1,536 dims), 1.19 GB
  - logits.parquet: 127,896 rows × 14,800 cols (5 metadata + 14,795 Perch taxonomy), 11.3 GB
  - No NaN/inf, embedding mean ~0, logit mean ~-1.3
  - Row counts match: 10,658 files × 12 chunks = 127,896 ✅
- [x] B.5 Verify train_audio outputs
  - embeddings.parquet: 265,924 rows × 1,541 cols (5 metadata + 1,536 dims), 2.4 GB
  - logits.parquet: 265,924 rows × 14,800 cols (5 metadata + 14,795 Perch taxonomy), 23.2 GB
  - No NaN/inf, embedding mean ~0.003, logit mean ~-1.1
  - All 35,549 files represented, row counts match between embeddings and logits ✅

## Phase C: Apply MLP Probes ← UNBLOCKED (both sources ready)

Artifacts: `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/`

**Both train_soundscapes and train_audio embeddings are ready.**

- [ ] C.0 Quick preview: compare existing `shared/embeddings/birdclef-2026/perch-v2/labeled/perch_mapped_scores.parquet`
  against v5j per-species AUC on 75 labeled-soundscape species (can do before Phase B completes)
- [ ] C.1 Load pretrained artifacts: `emb_scaler.joblib` → `pca_transformer.joblib` → `probe_models.joblib`
- [ ] C.2 Apply pipeline to Phase B embeddings: raw 1536-dim → StandardScaler → PCA(128) → 5-fold MLP probes → 234-class probs
- [ ] C.3 Save Perch 234-class probabilities to `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet`

## Phase D: Perch vs SED Comparative EDA

The key analysis — answers the 5 questions from EDA deep dive section 10.8:

- [ ] D.1 **Coverage**: How many of the 50 SED-zero species does Perch cover at >0.5?
- [ ] D.2 **Truly hard species**: How many of the 34 species (zero even at >0.3 with SED max-folds) does Perch cover?
- [ ] D.3 **Union coverage**: What's `max(SED-4fold) + Perch MLP probes` total species coverage at multiple thresholds?
- [ ] D.4 **Complementarity**: Venn diagram — Perch-only, SED-only, both, neither
- [ ] D.5 **Per-species agreement**: For species both detect, Spearman correlation per species (quality signal)
- [ ] D.6 **Per-site analysis**: Does Perch do better on the 14 unlabeled-only sites? (site diversity)
- [ ] D.7 **Ensemble teacher simulation**: What threshold/weight combo maximizes coverage?
- [ ] D.8 Update inference EDA doc and script with all findings

**Success criteria** (from proposal):
- Phase D succeeds if Perch+SED union covers ≥120/159 unseen species at >0.3
- If Perch covers 30+ of the 50 SED-zero species, ensemble teacher is justified

## Phase E: Ecological Sanity Check

- [ ] E.1 Review suspect species against Pantanal ecology
  - Original 54 species (≥50 focal, 0 soundscape predictions from single-fold)
  - Note: max-across-folds recovered some (House Sparrow, Swallow-tailed Manakin) — focus on
    species still at zero even with max-folds
- [ ] E.2 Categorize: genuinely absent / domain-shift failure / uncertain
- [ ] E.3 Document findings in results.md

## Synthesis

- [ ] F.1 Write final results.md with all findings and recommendations
- [ ] F.2 Design pseudo-labeling teacher strategy:
  - Which teacher: SED max-folds alone, Perch alone, or ensemble?
  - What threshold and power transform parameters?
  - Multi-label capture: threshold vs top-N vs soft labels?
  - Loss weighting: ground truth vs pseudo-label weights?
- [ ] F.3 Draft exp 025 proposal (pseudo-labeling pipeline)
