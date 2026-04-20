# Ridge submission (exp 106)

Pure linear model: `scores = perch_embedding @ W` where W is (1536, 234).

## Source

- **Weights**: `exp106/artifacts/ridge_model.npz` — config `focal_emb_empirical_bayes`, λ=100
- **Local metric**: 0.749 LOSO pooled AUROC (exp 106 CLAUDE.md)
- **No bias, no sigmoid** (raw `X @ W`; Kaggle macro AUROC is rank-based per species)

## Dry-run (validated 2026-04-20, job 6657296)

```bash
sbatch sbatch/dry_run_ridge.sbatch
```
- 3 training soundscapes → 36×235 CSV
- 34 s walltime on cpu-small
- score range [-0.098, 0.229]

## Kaggle packaging

### 1. Stage the two datasets locally

```bash
sbatch sbatch/stage_ridge.sbatch
# Writes:
#   artifacts/dist/deps/           -- wheels for onnxruntime/soundfile/numpy/pandas
#   artifacts/dist/weights-ridge/  -- perch_v2_no_dft.onnx + ridge_model.npz
#                                  + perch_species_mapping.json
#                                  + vendored birdclef_kaggle/*.py
```

### 2. Upload the datasets (user-run, network required)

```bash
# First time
kaggle datasets create -p artifacts/dist/deps
kaggle datasets create -p artifacts/dist/weights-ridge

# Subsequent updates
kaggle datasets version -p artifacts/dist/deps           -m "update"
kaggle datasets version -p artifacts/dist/weights-ridge  -m "update"
```

### 3. Push the kernel (user-run)

```bash
cd submissions/ridge/kaggle_kernel
kaggle kernels push
# Then attach to competition via the Kaggle UI or `kaggle kernels output`
```

Kernel metadata (`kaggle_kernel/kernel-metadata.json`):
- `is_private: true`
- `enable_internet: false`
- `enable_gpu: false`
- Datasets: `acmiyaguchi/birdclef-2026-deps`, `acmiyaguchi/birdclef-2026-weights-ridge`
- Competition: `birdclef-2026`

### 4. Result

LB score goes here once submitted.
