# Experiment 018: Perch Probes

Climb the Perch-only leaderboard ladder by progressively improving the probe
pipeline: OOF cross-validation → richer features → MLP probes → ProtoSSM → post-processing.

See [proposal.md](proposal.md) for hypothesis, motivation, and design rationale.
See [overview.md](overview.md) for a plain-language pipeline explanation.
See [tasks.md](tasks.md) for the full task checklist and phase breakdown.

**Branch**: `mg/018-perch-ensemble` (PR #20)
**Current Best**: 0.893 LB (Perch + LogReg probes, logit-space, no OOF)
**Target**: 0.925 LB (match yuriy/yaroslav)

## Leaderboard Reference Chain

| Score | Notebook | Key technique |
|-------|----------|---------------|
| 0.893 | **Ours (v1)** | Perch + LogReg probes, no OOF |
| 0.910 | chiranjithdharma | + OOF 5-fold GroupKFold, fold-safe priors |
| 0.913 | zirach | + ProtoSSM temporal modeling |
| 0.917 | pantanal-distill | + ResidualSSM boosting |
| 0.921 | maryna | + focal loss, mixup, cross-attn, SWA, TTA |
| 0.925 | yuriy/yaroslav | + wider SSM, 6-stage post-processing |

## Results Summary

| Submission | LB | Notes |
|------------|-----|-------|
| Perch raw (no probes) | 0.825 | Matches starter notebook |
| Perch cache v1 (genus proxy) | **0.836** | +0.011 from genus mapping |
| Perch probes v1 (logit-space) | **0.893** | +0.057 from probes, no OOF |
| Perch probes v2 (OOF) | pending | Phase C: OOF + fold-safe priors |

## Execution

### Push a submission

```bash
cd user/murilo/experiments/018-perch-probes

# v2: OOF fold-safe priors + Gaussian smoothing (current)
kaggle_kernel/push.sh perch-probes-v2

# v1: original logit-space pipeline (0.893 LB, for reference)
kaggle_kernel/push.sh perch-probes

# Check results
kaggle competitions submissions birdclef-2026 | head -20
```

### PACE: Offline preparation

```bash
cd user/murilo/experiments/018-perch-probes
sbatch sbatch/extract_embeddings.sh   # cpu-small, ~1hr
```

Individual steps:

```bash
python scripts/build_species_mapping.py       # → artifacts/perch_species_mapping.json
python scripts/extract_perch_embeddings.py    # → shared/embeddings/.../labeled/*.parquet
python scripts/train_probes.py                # → shared/artifacts/.../perch-ensemble-v1/
python scripts/build_artifacts.py             # → verify + dataset-metadata.json
python scripts/local_validate.py --sweep-weights  # → evaluate + find optimal blend
```

### Kaggle Kernel Data Sources

| Dataset | Content |
|---------|---------|
| `murilogustineli/birdclef-2026-perch-ensemble-v1` | Species mapping + config JSON |
| `murilogustineli/birdclef-2026-perch-labeled-embeddings` | Pre-extracted labeled embeddings (NPZ + parquet) |
| `murilogustineli/birdclef-tf220-wheel` | TensorFlow 2.20 wheels |
| `google/.../perch_v2_cpu/1` | Frozen Perch v2 model |

## Directory Structure

```
018-perch-probes/
├── README.md
├── overview.md                          # Plain-language pipeline explanation
├── proposal.md                          # Hypothesis, motivation, references
├── tasks.md                             # Phase C/D/E checklist
├── pyproject.toml
├── config/
│   └── perch_ensemble_v1.yaml           # All paths + hyperparameters
├── scripts/
│   ├── build_species_mapping.py         # Perch labels → BirdCLEF mapping (203/6/25)
│   ├── extract_perch_embeddings.py      # Run Perch on soundscapes
│   ├── build_labeled_embeddings_dataset.py  # Build NPZ for Kaggle upload
│   ├── train_probes.py                  # PCA + LogReg probes + metadata priors
│   ├── build_artifacts.py               # Verify + package for Kaggle upload
│   ├── local_validate.py                # Evaluate, sweep blend weights
│   └── upload_kaggle.py                 # Upload datasets to Kaggle
├── sbatch/
│   └── extract_embeddings.sh            # CPU job: chains offline steps
├── kaggle_kernel/
│   ├── push.sh                          # Push helper: ./push.sh perch-probes
│   ├── submission_perch_probes.py       # Main kernel: Perch + probes + submit
│   ├── submission_perch_cache.py        # Cache kernel (Perch inference only)
│   ├── submission_perch_raw.py          # Raw Perch baseline (no probes)
│   └── kernel-metadata-*.json           # Kaggle kernel metadata files
├── artifacts/
│   └── perch_species_mapping.json       # Species mapping (203 direct + 6 genus + 25 unmapped)
└── logs/                                # SLURM extraction logs

Shared storage (not in repo):
  shared/embeddings/birdclef-2026/perch-v2/
    └── labeled/                         # perch_embeddings.parquet, perch_logits.parquet
  shared/datasets/birdclef-2026/perch-labeled-embeddings/
    ├── labeled_perch_arrays.npz         # scores_raw + emb arrays
    └── labeled_perch_meta.parquet       # window metadata
```

## Key References

- [Perch Probes 0.910](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md) — primary reference
- [ProtoSSM 0.913](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill 0.917](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Maryna Optimized 0.921](../../docs/references/kaggle/birdclef-2026/notebooks/birdclef2026-optimized-maryna.md)
- [Yuriy ProtoSSM v5 0.925](../../docs/references/kaggle/birdclef-2026/notebooks/protossm-v5-time-optimised-yuriy.md)
- [Exp 017 (SED v4b)](../017-mel-overhaul/README.md)
