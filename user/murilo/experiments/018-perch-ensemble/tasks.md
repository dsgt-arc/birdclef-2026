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

### Step 1: Offline Pipeline (PACE)

Run all steps via a single sbatch job:

```bash
cd user/murilo/experiments/018-perch-ensemble
sbatch sbatch/extract_embeddings.sh   # cpu-small, ~1hr
```

Or run individually:

```bash
python scripts/build_species_mapping.py       # → artifacts/perch_species_mapping.json
python scripts/extract_perch_embeddings.py    # → shared/embeddings/.../labeled/*.parquet
python scripts/train_probes.py                # → shared/artifacts/.../perch-ensemble-v1/
python scripts/build_artifacts.py             # → verify + dataset-metadata.json
```

All scripts read paths from `config/perch_ensemble_v1.yaml`.

- [ ] 1.1 Run species mapping — verify 203 direct, 6 genus proxy, 25 unmapped
- [ ] 1.2 Run Perch extraction — verify embeddings (792, 1536), mapped scores (792, 234)
- [ ] 1.3 Run probe training — check CV AUC summary, verify artifacts in shared storage
- [ ] 1.4 Run build_artifacts — verify all files present + dataset-metadata.json created

### Step 2: Local Validation (Perch-only)

```bash
python scripts/local_validate.py
```

- [ ] 2.1 Record incremental metrics: raw → +probes → +priors → +smoothing
- [ ] 2.2 Compare local AUROC to chiranjithdharma's expected ~0.910

### Step 3: Perch-Only Kaggle Submission

Kernel uses `PERCH_ONLY = True` (already set in submission.py).
Uses `kernel-metadata-perch-only.json` (no SED weights or birdclef-deps needed).

```bash
# Upload probe artifacts as Kaggle dataset
PROBE_DIR="/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
cd "$PROBE_DIR"
kaggle datasets create -p . --dir-mode zip

# Push Perch-only kernel
cd user/murilo/experiments/018-perch-ensemble/kaggle_kernel
kaggle kernels push -p . --metadata kernel-metadata-perch-only.json

# Check results
kaggle kernels status murilogustineli/birdclef-2026-perch-only-v1
kaggle competitions submissions birdclef-2026 | head -20
```

- [ ] 3.1 Upload probe artifacts to Kaggle
- [ ] 3.2 Push Perch-only kernel
- [ ] 3.3 Verify submission completes within 90min budget
- [ ] 3.4 Record Perch-only LB result

### Phase A Decision Gate

- If Perch-only LB >= 0.900: pipeline works → continue to Phase B (ensemble)
- If Perch-only LB 0.850-0.900: partially works → debug probes/priors before ensemble
- If Perch-only LB < 0.850: something is wrong → fix before proceeding

---

## Phase B: SED + Perch Ensemble

Only proceed after Phase A Perch-only baseline is validated.

### Step 4: Local Ensemble Validation

```bash
python scripts/local_validate.py --sed-predictions <path> --sweep-weights
```

- [ ] 4.1 Generate SED predictions on labeled soundscapes (or use existing exports)
- [ ] 4.2 Run local validation with weight sweep
- [ ] 4.3 Record optimal blend weights and per-taxa breakdown
- [ ] 4.4 Compare ensemble AUROC vs Perch-only AUROC vs SED-only AUROC

### Step 5: Ensemble Kaggle Submission

Set `PERCH_ONLY = False` in submission.py, update blend weights from sweep.
Uses `kernel-metadata-ensemble-v1.json` (includes SED weights + birdclef-deps).

```bash
# Push ensemble kernel
cd user/murilo/experiments/018-perch-ensemble/kaggle_kernel
kaggle kernels push -p . --metadata kernel-metadata-ensemble-v1.json

# Check results
kaggle kernels status murilogustineli/birdclef-2026-perch-ensemble-v1
kaggle competitions submissions birdclef-2026 | head -20
```

- [ ] 5.1 Set `PERCH_ONLY = False`, update blend weights in submission.py
- [ ] 5.2 Push ensemble kernel
- [ ] 5.3 Verify submission completes within 90min budget
- [ ] 5.4 Record ensemble LB result vs Perch-only and SED baselines

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
