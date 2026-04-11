# Experiment 025: Iterative Pseudo-Labeling — Tasks

## Goal

Close the LB gap (0.929 → 0.948) by training v5j on focal + labeled soundscapes +
pseudo-labeled soundscapes, using a tiered teacher: SED max-folds for 184 species where
SED has good predictions, and Perch with per-species z-normalization for 44 species where
SED is blind.

Status: **Phase A complete. Phase B.1 complete. Round 1 training done, submitted to LB — awaiting score.**

**Branch**: `mg/025-iter-pseudo-labeling`

---

## Phase A: Generate Pseudo-Labels ✅

Script: `generate_pseudo_labels.py` (config-driven, reads `configs/v1.yaml`)
Output: `shared/pseudo-labels/birdclef-2026/v1/`

### A.1: SED pseudo-labels (Tiers A+B — 184 species) ✅

- [x] Write `generate_pseudo_labels.py` (config-driven, versioned output)
- [x] Load SED 4-fold logits → sigmoid → max-across-folds
- [x] Apply temperature scaling (T=0.6) → power transform (P^1.5)
- [x] Save `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet`
- [x] Verify: 198K pseudo-labels >0.5, 162/184 species covered

### A.2: Perch pseudo-labels (Tier C — 44 SED-zero species) ✅

- [x] Identify 44 Tier C species (unseen, SED max <0.5)
- [x] Per-species z-normalization: mean/std across 127K segments
- [x] 1 species skipped (1595929, std=0.0097 < 0.01 threshold)
- [x] Z > 2.0 → sigmoid → P^1.5 → 43 active species
- [x] Save `shared/pseudo-labels/birdclef-2026/v1/perch_pseudo_labels.parquet`
- [x] Verify: 149K pseudo-labels >0.5, 34/43 species covered, median 3,728 segments/species

### A.3: Merge and validate ✅

- [x] Merged → `shared/pseudo-labels/birdclef-2026/v1/pseudo_labels.parquet`
- [x] **347K pseudo-labels >0.5** across **196/234 species**, **2.73 species/segment** average
- [x] Tier assignment saved to `tier_assignment.csv`, stats to `stats.json`
- [x] Config copied to output dir for reproducibility

## Phase B: Train Student (Round 1)

### B.1: Extend data loader ✅

Shared src modifications (backward compatible with exp 021):

- [x] `src/birdclef/data/sed.py`: Add `PseudoLabelSoundscapeDataset` (loads OGG + soft
  pseudo-labels from parquet, returns 4-tuple with loss_weight)
- [x] `src/birdclef/data/sed.py`: Add `GTDatasetWrapper` (wraps 3-tuple → 4-tuple with w=1.0)
- [x] `src/birdclef/data/sed.py`: Add `StratifiedBatchSampler` (50/50 GT vs PL per batch)
- [x] `src/birdclef/data/sed.py`: Extend `SEDDataModule` with `pseudo_label_path`,
  `pseudo_label_weight`, `pseudo_label_batch_fraction` params + fold leakage prevention
- [x] `src/birdclef/lightning/sed_finetune.py`: Per-sample loss weighting in `_compute_loss`
  (BCE path, `reduction='none'` → weight → mean)
- [x] `src/birdclef/lightning/sed_finetune.py`: Pass `loss_weight` through `_compute_chunked_loss`,
  `training_step` (4-tuple unpack), `_apply_mixup`, `_apply_manifold_mixup`

Exp 025 files:

- [x] `train.py`: Copied from exp 021, added `--pseudo-label-path` arg, updated experiment name
- [x] `configs/v1_train.yaml`: v5j recipe + pseudo-label data config (BCE, w=0.5)
- [x] `sbatch/_job.sh` + `run.sh`: SLURM scripts with `PSEUDO_LABEL_PATH` forwarding
- [x] `fold_assignments.json`: Copied from exp 021 (same 4-fold splits)

### B.2: Train and evaluate

- [x] Fast-dev-run sanity check ✅ — 5 batches, 32724 GT + 127896 PL = 160620 total,
  50/50 batch composition confirmed, loss_weight=0.5 for PL, train/loss=0.529
- [x] GPU-check (1 epoch) ✅ — memory and throughput OK on RTX 6000
- [x] Full training run (90/10 split) ✅ — 20 epochs, best val/auroc=0.872 (epoch 18),
  val/cmap=0.620, train/loss=0.015. Checkpoint: `best-epochepoch=18.ckpt`
  SLURM job 6401689, RTX 6000, 1h08m walltime.
- [x] ONNX export + Kaggle submission pushed ✅ — `hgnet-b0-sed-v5j-pl-v1`, 22.4s inference
- [ ] **Awaiting LB score** → compare against v5j baseline (0.903-0.906)
- [ ] **Decision gate**: If LB > 0.906 → proceed to 4-fold. If not → diagnose and adjust.
- [ ] Train 4-fold ensemble (if single model improves)
- [ ] Submit 4-fold ensemble (compare against v5j 4-fold = 0.913)
- [ ] Submit student + Perch ensemble (compare against 0.929 best)

### B.3: Analyze results

- [ ] Per-species analysis: which unseen species improved? Any regressions on seen species?
- [ ] Check labeled soundscape AUC (should remain >0.99)
- [ ] Compare student predictions on soundscapes vs teacher — is student more calibrated?

## Phase C: Iterate (conditional on Phase B success)

### C.1: Round 2 (if round 1 gains on LB)

- [ ] Run round 1 student inference on all soundscapes
- [ ] Re-generate pseudo-labels using student as teacher
- [ ] Student should be better calibrated for Tier C species (trained on soundscapes)
- [ ] Consider relaxing Perch z-threshold (2.0 → 1.5) for more coverage
- [ ] Train round 2 student, submit to LB

### C.2: Variants (if round 1 needs adjustment)

- [ ] Try PL loss weight 0.3 (if too noisy) or 0.7 (if too conservative)
- [ ] Try power transform P^2.0 (more noise suppression)
- [ ] Try Perch z-threshold 1.5 or 2.5
- [ ] Try SED temperature 0.5 or 0.9
- [ ] Try leave-one-fold-out pseudo-labels (if fold leakage is suspected):
  fold K trains on max(other 3 folds) only — cleaner but slightly less coverage

## Success Criteria

- [ ] Round 1 single model (90/10): **>0.906** LB (v5j baseline)
- [ ] Round 1 4-fold ensemble: **>0.913** LB (v5j ensemble)
- [ ] Round 1 + Perch ensemble: **>0.929** LB (current best)
- [ ] Stretch: **>0.935** LB
