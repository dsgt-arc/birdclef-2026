# Experiment 025: Pseudo-Labeling v2 — Tasks

## Goal

Close the LB gap (0.929 → 0.948) by adding SED-generated pseudo-labels for 127K unlabeled
soundscape segments, providing same-domain training data for 159 unseen species.

Status: **Proposed — pending exp 024 Phase F completion and review.**

---

## Phase A: Generate Pseudo-Labels

- [ ] A.1 Write `scripts/generate_pseudo_labels.py`:
  - Load SED 4-fold logits from `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/`
  - Compute max(fold0..fold3) per species per segment
  - Apply P^1.5 power transform
  - Threshold at >0.5
  - Save to `shared/pseudo-labels/v1/pseudo_labels.parquet`
- [ ] A.2 Verify pseudo-label statistics:
  - Species coverage at multiple thresholds
  - Pseudo-labels per species distribution
  - Sanity check: seen species should dominate

## Phase B: Train Student (Round 1)

- [ ] B.1 Extend SED data loader to include pseudo-labeled soundscapes
  - Add pseudo-label parquet as additional training source
  - Implement stratified batch sampling (50% focal/labeled, 50% pseudo)
  - Add per-source loss weighting (w=0.5 for pseudo-labels)
- [ ] B.2 Train v5j + pseudo-labels (fold 0 first as sanity check)
- [ ] B.3 Submit fold 0 to LB — compare against v5j fold 0 baseline (0.906)
- [ ] B.4 If fold 0 shows promise: train remaining 3 folds
- [ ] B.5 Submit 4-fold ensemble to LB — compare against v5j 4-fold (0.913)
- [ ] B.6 Submit student + Perch ensemble — compare against 0.929 best

## Phase C: Evaluate and Iterate (conditional)

- [ ] C.1 Analyze: which species improved? Which regressed?
- [ ] C.2 If gain: student-as-teacher → round 2 pseudo-labels
- [ ] C.3 If gain: consider Perch Tier C (40 species, top-K filtering)
- [ ] C.4 If regression: diagnose (threshold? weight? recipe sensitivity?)

## Variants (conditional on Phase B success)

- [ ] V.1 Higher threshold (>0.7)
- [ ] V.2 Lower PL weight (0.3)
- [ ] V.3 Power 2.0
- [ ] V.4 Perch Tier C (40 species)
