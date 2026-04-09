# Experiment 024: Data-Centric Exploration — Tasks

## Goal

Understand model coverage gaps and Perch/SED complementarity to inform the pseudo-labeling
teacher strategy for exp 025.

Status: **Phase A complete. Phase B next (Perch TF inference).**

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

## Phase B: Perch v2 TF Inference

- [x] B.1 Write standalone `inference/perch_inference.py` (TF SavedModel, saves embeddings + logits)
- [x] B.2 Write SLURM batch scripts (`inference/sbatch/perch_inference.sh` + `run_perch.sh`)
- [ ] B.3 Run inference → save embeddings + raw logits to `shared/inference/birdclef-2026/perch-v2-tf/`
  - Submit: `cd user/murilo/inference && bash sbatch/run_perch.sh --source train_soundscapes`
  - Estimated: ~21h CPU for 127K soundscape segments
- [ ] B.4 Verify output shapes and sample predictions

## Phase C: Apply MLP Probes

- [ ] C.1 Load pretrained artifacts (scaler, PCA, probes) from `shared/artifacts/020-perch-onnx/`
- [ ] C.2 Apply pipeline: raw embeddings → scale → PCA → MLP probes → 234-class predictions
- [ ] C.3 Save Perch 234-class probabilities to parquet

## Phase D: Perch vs SED Comparative EDA

- [ ] D.1 Species coverage comparison (Perch vs SED vs union at multiple thresholds)
- [ ] D.2 Complementarity analysis (Perch-only, SED-only, both, neither)
- [ ] D.3 Per-species agreement (Spearman correlation for shared species)
- [ ] D.4 Per-site analysis (does Perch do better on 14 unlabeled-only sites?)
- [ ] D.5 Ensemble coverage simulation (what threshold/weight maximizes coverage?)
- [ ] D.6 Update inference EDA doc with all findings

## Phase E: Ecological Sanity Check

- [ ] E.1 Review 54 suspect species (high focal, zero soundscape) against Pantanal ecology
- [ ] E.2 Categorize: genuinely absent / domain-shift failure / uncertain
- [ ] E.3 Document findings in results.md

## Synthesis

- [ ] F.1 Write results.md with all findings and recommendations
- [ ] F.2 Design pseudo-labeling teacher strategy based on Perch+SED complementarity
- [ ] F.3 Draft exp 025 proposal (pseudo-labeling pipeline)
