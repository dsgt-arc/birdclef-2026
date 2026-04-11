# Experiment 025: Pseudo-Labeling v2 — Tasks

## Goal

Close the LB gap (0.929 → 0.948) by training v5j on focal + labeled soundscapes +
pseudo-labeled soundscapes, using a tiered teacher: SED max-folds for 184 species where
SED has good predictions, and Perch with per-species z-normalization for 44 species where
SED is blind.

Status: **Proposed — awaiting review.**

**Branch**: TBD

---

## Phase A: Generate Pseudo-Labels

### A.1: SED pseudo-labels (Tiers A+B — 184 species)

- [ ] Write `scripts/generate_pseudo_labels.py`
- [ ] Load SED 4-fold logits → sigmoid → max-across-folds
- [ ] Apply temperature scaling (T=0.6) → power transform (P^1.5)
- [ ] Save `shared/pseudo-labels/v1/sed_pseudo_labels.parquet`
- [ ] Verify: coverage stats, per-species segment counts, sanity checks

### A.2: Perch pseudo-labels (Tier C — 44 SED-zero species)

- [ ] Identify the 44 Tier C species (unseen, SED max <0.5, Perch raw >0.3)
- [ ] For each: compute mean/std of Perch raw prob across 127K segments
- [ ] Skip species with std < 0.01 (no Perch signal variation)
- [ ] Z-normalize per species → keep z > 2.0 → sigmoid(z) → P^1.5
- [ ] Save `shared/pseudo-labels/v1/perch_pseudo_labels.parquet`
- [ ] Verify: per-species segment counts, z-score distributions

### A.3: Merge and validate

- [ ] Merge SED + Perch labels → `shared/pseudo-labels/v1/pseudo_labels.parquet`
  - Tiers A+B: SED labels
  - Tier C: Perch labels
  - Tier D: zeros (6 invisible species)
- [ ] Summary statistics: species coverage, segments per species, label density
- [ ] Sanity: no overlap between SED and Perch tiers

## Phase B: Train Student (Round 1)

### B.1: Extend data loader

- [ ] Modify SED training pipeline to load pseudo-label parquet
- [ ] Implement stratified batch sampling (50% focal/labeled, 50% pseudo-labeled)
- [ ] Add per-source loss weighting (w=1.0 ground truth, w=0.5 pseudo-labels)
- [ ] Soft label support in BCE loss (pseudo-labels are probabilities, not 0/1)

### B.2: Train and evaluate

- [ ] Train fold 0 (sanity check — compare against v5j fold 0 = 0.906)
- [ ] Submit fold 0 to LB
- [ ] **Decision gate**: If fold 0 > 0.906 → proceed. If not → diagnose and adjust.
- [ ] Train folds 1-3
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

## Success Criteria

- [ ] Round 1 fold 0: **>0.906** LB (v5j baseline)
- [ ] Round 1 4-fold: **>0.913** LB (v5j ensemble)
- [ ] Round 1 + Perch ensemble: **>0.929** LB (current best)
- [ ] Stretch: **>0.935** LB
