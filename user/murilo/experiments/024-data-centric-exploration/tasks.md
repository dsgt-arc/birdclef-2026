# Experiment 024: Data-Centric Exploration — Tasks

## Goal

Understand model coverage gaps and Perch/SED complementarity to inform the pseudo-labeling
teacher strategy for exp 025.

Status: **Phase A complete. Phase B complete. Phase C complete (train_soundscapes). Phase D unblocked.**

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

## Phase C: Apply MLP Probes ✅

Artifacts: `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/`
Script: `user/murilo/scripts/apply_perch_probes.py`

**Key discovery**: The MLP probes require **142 features** (128 PCA + 14 contextual), not
just PCA embeddings. The full pipeline from the exp 023 submission kernel was extracted:
Perch logit→BirdCLEF mapping → prior fusion → temporal smoothing → MLP probes → alpha blend.

- [x] C.0 Verified Phase B embeddings match exp 020 probe training data
  - Cosine similarity = 1.000000 between Phase B and exp 018/020 labeled embeddings
  - Same Perch v2 TF SavedModel → identical embedding space → no retraining needed
- [x] C.1 Wrote `apply_perch_probes.py` extracting full Perch→234 pipeline from exp 023 kernel
  - Perch 14,795 logits → 234 BirdCLEF mapping (203 direct, 3 genus proxy, 28 unmapped sonotypes)
  - Prior fusion (site/hour tables), temporal smoothing (texture + event)
  - 58 MLP probes with 142-dim features: PCA(128) + raw/prior/base logits + temporal + interactions
  - Alpha blend (0.45) with base scores
  - Memory-optimized: reads only 220/14,795 logit columns (avoids OOM on login node)
- [x] C.2 Applied pipeline to train_soundscapes (127,896 segments, 62 seconds on CPU)
- [x] C.3 Saved three outputs to `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/`:
  - `perch_234_raw_probs.parquet` — pure Perch signal (mapped logits + genus proxy, sigmoid), 158 MB
  - `perch_234_scores.parquet` — full pipeline logit-space (+ priors, smoothing, probes), 172 MB
  - `perch_234_probs.parquet` — full pipeline probability-space, 171 MB
- [x] C.4 Verified outputs: no NaN/inf, correct shapes, distribution sanity

**Coverage preview** (full pipeline):
- 212/234 species at >0.5 (vs SED max-folds: 109/159 unseen at >0.5)
- 201/234 species at >0.9
- 28 sonotypes have no Perch mapping (rely entirely on priors)

**Raw Perch coverage** (before priors/probes — fair comparison with SED):
- 195/234 species at >0.5
- 28 true zeros (unmapped sonotypes)

**Note**: Full pipeline suppresses unmapped inactive species to -8.0 (sigmoid ≈ 0.0003),
discarding valid genus proxy signal for some amphibia. Phase D should use raw_probs for
fair detection comparison.

## Phase D: Perch vs SED Comparative EDA ✅

Script: `user/murilo/scripts/perch_sed_comparison.py`
Output: `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` sections 12-13

- [x] D.1 **Coverage**: Perch rescues **40** SED-zero species at >0.5 (109→149/159)
- [x] D.2 **Truly hard species**: Perch covers **28/34** at >0.3, **24/34** at >0.5. Only 6 invisible.
- [x] D.3 **Union coverage**: **153/159** unseen at >0.3 ✓ PASS (target: ≥120)
- [x] D.4 **Complementarity**: **Zero SED-only species** at any threshold. 28 Perch-only at >0.3.
- [x] D.5 **Per-species agreement**: Spearman ρ mean=0.223 — models are highly complementary
- [x] D.6 **Per-site analysis**: Perch ~145/159 per site vs SED ~13/159. Equal on labeled/unlabeled sites.
- [x] D.7 **Ensemble teacher**: SED=5K pseudo-labels, Perch=15M (!). **Perch calibration problem.**
- [x] D.8 Updated EDA doc sections 12-13 with all findings + calibration analysis

**Success criteria**: Both ✓ PASS
- ✅ Union ≥120/159 unseen at >0.3: **153/159**
- ✅ Perch covers 30+ of 50 SED-zero: **40**

**CRITICAL finding**: Perch raw probabilities are severely miscalibrated (15M pseudo-labels
vs SED's 5K). Perch predicts most species in most segments. Must use tiered teacher:
SED-primary + Perch-rescue for SED-blind species with aggressive filtering.

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
