# Experiment 018: Perch Ensemble

Ensemble EfficientNet-B0 SED (v4b, 0.859 LB) with Google Perch v2 foundation model.
See [proposal.md](proposal.md) for hypothesis, motivation, and design rationale.

**Branch**: `mg/018-perch-ensemble`
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

## Data Paths (Verified)

| Resource | Path | Verified |
|----------|------|----------|
| Perch v2 SavedModel | `/storage/project/.../shared/models/perch-v2/` | saved_model.pb + variables/ |
| Perch labels | `.../perch-v2/assets/labels.csv` | 14,796 entries (line 0 = `inat2024_fsd50k`) |
| BirdCLEF taxonomy | `.../shared/data/birdclef-2026/taxonomy.csv` | 234 species (primary_label = inat_taxon_id) |
| Soundscape labels | `.../shared/data/birdclef-2026/train_soundscapes_labels.csv` | 1,478 rows, 66 files, 75 species, semicolon-separated IDs |
| Soundscape audio | `.../shared/data/birdclef-2026/train_soundscapes/` | 10,658 files total (66 labeled) |

## Execution

### PACE: Offline preparation (single sbatch job)

```bash
cd user/murilo/experiments/018-perch-ensemble
sbatch sbatch/extract_embeddings.sh   # cpu-small, ~1hr
```

Chains: species mapping → Perch extraction → probe training → artifact packaging.

Individual steps:

```bash
python scripts/build_species_mapping.py       # → artifacts/perch_species_mapping.json
python scripts/extract_perch_embeddings.py    # → outputs/*.parquet (792 rows)
python scripts/train_probes.py                # → artifacts/perch_probes/
python scripts/build_artifacts.py             # → verify + dataset-metadata.json
python scripts/local_validate.py --sweep-weights  # → evaluate + find optimal blend
```

### Kaggle: Upload + submit

```bash
# Upload probe artifacts as dataset
cd artifacts/perch_probes
kaggle datasets create -p . --dir-mode zip

# Push kernel
cd ../../kaggle_kernel
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
├── proposal.md                          # Hypothesis, motivation, references
├── tasks.md                             # Step-by-step checklist
├── pyproject.toml                       # Dependencies (tensorflow, sklearn, etc.)
├── config/
│   └── perch_ensemble_v1.yaml           # All hyperparameters
├── scripts/
│   ├── build_species_mapping.py         # Perch labels → BirdCLEF mapping (203/6/25)
│   ├── extract_perch_embeddings.py      # Run Perch on 66 labeled soundscapes
│   ├── train_probes.py                  # PCA + LogReg probes + metadata priors
│   ├── build_artifacts.py               # Verify + package for Kaggle upload
│   └── local_validate.py               # Evaluate ensemble, sweep blend weights
├── sbatch/
│   └── extract_embeddings.sh            # CPU job: chains all offline steps
├── kaggle_kernel/
│   ├── submission.py                    # Unified SED + Perch kernel
│   └── kernel-metadata-ensemble-v1.json # Kaggle kernel metadata
├── artifacts/
│   ├── perch_species_mapping.json       # Species mapping output
│   └── perch_probes/                    # Serialized probes, PCA, priors
└── outputs/
    ├── perch_embeddings.parquet         # 792 rows × 1536 dims
    ├── perch_logits.parquet             # 792 rows × 14796 dims
    └── perch_mapped_scores.parquet      # 792 rows × 234 species
```

## Key References

- [Perch Probes (0.910)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md) — primary reference
- [Perch ProtoSSM (0.913)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill (0.917)](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Gap analysis](../../notes/gap-analysis-2026-03-26.md)
- [Exp 017 (SED v4b)](../017-mel-overhaul/README.md)
- [Exp 010 (submission pipeline)](../010-submission-pipeline/README.md)
