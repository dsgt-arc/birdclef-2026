# Experiment 018: Perch Ensemble — Tasks

## Goal

First reproduce chiranjithdharma's Perch-only pipeline (~0.910 LB) as a validated
baseline, then ensemble with SED v4b (0.859 LB) to see if combining fundamentally
different models pushes past 0.92.

**SED Baseline**: v4b = 0.859 LB
**Perch Baseline**: chiranjithdharma = 0.910 LB
**Target**: 0.92+ LB (ensemble)

### Status: IN PROGRESS

- **Branch**: `mg/018-perch-ensemble` (PR #20)

---

## Phase A: Perch-Only Baseline (reproduce chiranjithdharma ~0.910)

### Step 1: Species Mapping

- [ ] 1.1 Run `scripts/build_species_mapping.py`
- [ ] 1.2 Verify: 203 direct, 6 genus proxy, 25 unmapped insect sonotypes
- [ ] 1.3 Inspect genus proxy species (which amphibians? which genus matches?)

### Step 2: Extract Perch Embeddings

- [ ] 2.1 Submit sbatch job: `sbatch sbatch/extract_embeddings.sh`
- [ ] 2.2 Verify embeddings shape: (792, 1536) — 66 files × 12 windows
- [ ] 2.3 Verify mapped scores shape: (792, 234)
- [ ] 2.4 Sanity check: mapped_nonzero should be ~209 species (203 direct + 6 proxy)

### Step 3: Train Probes + Priors

- [ ] 3.1 Verify probe training completes (PCA, ~200 LogReg models, priors)
- [ ] 3.2 Check CV AUC summary (mean, median, top/bottom species)
- [ ] 3.3 Verify artifacts serialized in `artifacts/perch_probes/`
- [ ] 3.4 Run `scripts/build_artifacts.py` to verify + package

### Step 4: Local Validation (Perch-only)

- [ ] 4.1 Run `scripts/local_validate.py` (no --sed-predictions)
- [ ] 4.2 Record incremental metrics: raw → +probes → +priors → +smoothing
- [ ] 4.3 Compare local AUROC to chiranjithdharma's expected ~0.910

### Step 5: Perch-Only Kaggle Submission

- [ ] 5.1 Create Perch-only kernel variant (disable SED in submission.py)
- [ ] 5.2 Upload probe artifacts as Kaggle dataset
- [ ] 5.3 Push Perch-only kernel to Kaggle
- [ ] 5.4 Verify submission completes within 90min budget
- [ ] 5.5 Record Perch-only LB result

### Step 5 Decision Gate

- If Perch-only LB >= 0.900: pipeline works → continue to Phase B (ensemble)
- If Perch-only LB 0.850-0.900: partially works → debug probes/priors before ensemble
- If Perch-only LB < 0.850: something is wrong → fix before proceeding

---

## Phase B: SED + Perch Ensemble

Only proceed after Perch-only baseline is validated.

### Step 6: Local Ensemble Validation

- [ ] 6.1 Generate SED predictions on labeled soundscapes (or use existing exports)
- [ ] 6.2 Run `scripts/local_validate.py --sed-predictions <path> --sweep-weights`
- [ ] 6.3 Record optimal blend weights and per-taxa breakdown
- [ ] 6.4 Compare ensemble AUROC vs Perch-only AUROC vs SED-only AUROC

### Step 7: Ensemble Kaggle Submission

- [ ] 7.1 Update kernel with optimal blend weights from sweep
- [ ] 7.2 Push ensemble kernel to Kaggle
- [ ] 7.3 Verify submission completes within 90min budget
- [ ] 7.4 Record ensemble LB result vs Perch-only and SED baselines

---

## Results Tracking

| Submission | LB | vs SED (0.859) | vs Perch chiranjithdharma (0.910) |
|------------|-----|----------------|----------------------------------|
| Perch-only (ours) | pending | | |
| Ensemble v1 | pending | | |

---

## Key Config: v1

| Parameter | Value |
|-----------|-------|
| Perch model | `google/bird-vocalization-classifier/TensorFlow2/perch_v2/1` |
| PCA components | 32 |
| LogReg C | 0.25 |
| Min positive samples | 8 |
| Probe alpha base | 0.25 |
| Lambda event (Aves) | 0.4 |
| Lambda texture (Amphibia/Insecta) | 1.0 |
| Smoothing alpha event | 0.15 |
| Smoothing alpha texture | 0.35 |
| SED weight | 0.3 |
| Perch weight | 0.7 |
| Temperature | 1.10 |
