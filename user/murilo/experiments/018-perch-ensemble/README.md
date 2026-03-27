# Experiment 018: Perch Ensemble

Ensemble EfficientNet-B0 SED (v4b, 0.859 LB) with Google Perch v2 foundation model.
See [proposal.md](proposal.md) for hypothesis, motivation, and design rationale.
See [overview.md](overview.md) for a plain-language pipeline explanation.

**Branch**: `mg/018-perch-ensemble` (PR #20)
**SED Baseline**: Exp 017 v4b = 0.859 LB
**Perch Baseline**: chiranjithdharma = 0.910 LB
**Target**: 0.92+ LB
**Status**: Implementation complete, awaiting execution

## Results Summary

| Variant | SED weight | Perch weight | Local AUROC | LB |
|---------|-----------|-------------|-------------|-----|
| SED only (v4b) | 1.0 | 0.0 | — | 0.859 |
| Perch only | 0.0 | 1.0 | pending | pending |
| Ensemble v1 | 0.3 | 0.7 | pending | pending |

## Storage Layout

All paths are centralized in `config/perch_ensemble_v1.yaml`.

### Input Data (shared, read-only)

| Resource | Path | Details |
|----------|------|---------|
| Perch v2 model | `.../shared/models/perch-v2/` | TF SavedModel (saved_model.pb + variables/) |
| Perch labels | `.../shared/models/perch-v2/assets/labels.csv` | 14,796 entries |
| Taxonomy | `.../shared/data/birdclef-2026/taxonomy.csv` | 234 species |
| Soundscape labels | `.../shared/data/birdclef-2026/train_soundscapes_labels.csv` | 1,478 rows, 66 files |
| Soundscape audio | `.../shared/data/birdclef-2026/train_soundscapes/` | 10,658 files |

### Output Data (shared, written by scripts)

| Output | Path | Details |
|--------|------|---------|
| Perch embeddings | `.../shared/embeddings/birdclef-2026/perch-v2/labeled/` | Parquets: embeddings, logits, mapped scores (792 rows) |
| Probe artifacts | `.../shared/artifacts/birdclef-2026/perch-ensemble-v1/` | PCA, ~200 LogReg models, priors, alphas (uploaded to Kaggle) |

### Local Experiment Artifacts (committed to repo)

| Output | Path | Details |
|--------|------|---------|
| Species mapping | `artifacts/perch_species_mapping.json` | 203 direct + 6 genus + 25 unmapped |

## Execution

### PACE: Offline preparation (single sbatch job)

```bash
cd user/murilo/experiments/018-perch-ensemble
sbatch sbatch/extract_embeddings.sh   # cpu-small, ~1hr
```

Chains: species mapping → Perch extraction → probe training → artifact packaging.
All scripts read paths from `config/perch_ensemble_v1.yaml`.

Individual steps:

```bash
python scripts/build_species_mapping.py       # → artifacts/perch_species_mapping.json
python scripts/extract_perch_embeddings.py    # → shared/embeddings/.../labeled/*.parquet
python scripts/train_probes.py                # → shared/artifacts/.../perch-ensemble-v1/
python scripts/build_artifacts.py             # → verify + dataset-metadata.json
python scripts/local_validate.py --sweep-weights  # → evaluate + find optimal blend
```

### Kaggle: Upload + submit

```bash
# Upload probe artifacts as Kaggle dataset
PROBE_DIR="/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
cd "$PROBE_DIR"
kaggle datasets create -p . --dir-mode zip

# Push kernel
cd user/murilo/experiments/018-perch-ensemble/kaggle_kernel
kaggle kernels push -p .

# Check results
kaggle kernels status murilogustineli/birdclef-2026-perch-ensemble-v1
kaggle competitions submissions birdclef-2026 | head -20
```

### Kaggle Kernel Data Sources

| Dataset | Content |
|---------|---------|
| `murilogustineli/birdclef-deps` | onnxruntime wheels |
| `murilogustineli/birdclef-2026-effnet-b0-sed-v4b-weights` | SED ONNX model + mel config |
| `murilogustineli/birdclef-2026-perch-ensemble-v1` | Probe artifacts (PCA, models, priors) |
| `google/bird-vocalization-classifier/TensorFlow2/perch_v2/1` | Frozen Perch v2 (Kaggle model source) |

## Directory Structure

```
018-perch-ensemble/
├── README.md
├── overview.md                          # Plain-language pipeline explanation
├── proposal.md                          # Hypothesis, motivation, references
├── tasks.md                             # Phase A/B checklist
├── pyproject.toml
├── config/
│   └── perch_ensemble_v1.yaml           # All paths + hyperparameters
├── scripts/
│   ├── build_species_mapping.py         # Perch labels → BirdCLEF mapping (203/6/25)
│   ├── extract_perch_embeddings.py      # Run Perch on soundscapes (labeled/unlabeled)
│   ├── train_probes.py                  # PCA + LogReg probes + metadata priors
│   ├── build_artifacts.py               # Verify + package for Kaggle upload
│   └── local_validate.py               # Evaluate ensemble, sweep blend weights
├── sbatch/
│   └── extract_embeddings.sh            # CPU job: chains all offline steps
├── kaggle_kernel/
│   ├── submission.py                    # Unified SED + Perch kernel
│   └── kernel-metadata-ensemble-v1.json
└── artifacts/
    └── perch_species_mapping.json       # Species mapping (lightweight, in repo)

Shared storage (not in repo):
  shared/embeddings/birdclef-2026/perch-v2/
    └── labeled/                         # perch_embeddings.parquet, perch_logits.parquet, perch_mapped_scores.parquet
  shared/artifacts/birdclef-2026/perch-ensemble-v1/
    ├── pca_transformer.pkl              # PCA model
    ├── probe_models.pkl                 # ~200 LogReg models
    ├── probe_alphas.json                # Dynamic alpha per species
    ├── metadata_priors.json             # Site/hour Bayesian priors
    ├── taxa_mapping.json                # Species → taxa class
    ├── config.json                      # Ensemble config snapshot
    └── perch_species_mapping.json       # Copy for Kaggle upload
```

## Key References

- [Perch Probes (0.910)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md) — primary reference
- [Perch ProtoSSM (0.913)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill (0.917)](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Gap analysis](../../notes/gap-analysis-2026-03-26.md)
- [Exp 017 (SED v4b)](../017-mel-overhaul/README.md)
- [Exp 010 (submission pipeline)](../010-submission-pipeline/README.md)
