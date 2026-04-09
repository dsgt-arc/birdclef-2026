# Experiment 106: Perch v2 Bayesian Label Transfer

## What this experiment does

Extracts Perch v2 ONNX embedding (1536-d) and label logits (14,795-d) for all
BirdCLEF 2026 audio, then applies statistical methods to transfer Perch's
pretrained taxonomy to BirdCLEF 2026's 234 targets. Uses a star schema
framework where Perch logit space is the shared dimension connecting all
entities (species, sites, time periods, windows).

## Status

Phase 1 (static baselines) COMPLETE. Phase 2 in progress.
Best model: focal embedding ridge + empirical Bayes prior → **0.749 AUROC** (LOSO CV).
Multi-prototype Leiden clustering + site-state prototypes: running.

## Extracted data

All on scratch at `~/scratch/birdclef-2026/features/exp106/`:

| Source | Path | Rows | Size |
|--------|------|------|------|
| Labeled soundscapes | `soundscape_labeled.parquet` | 739 | 43 MB |
| Unlabeled soundscapes | `soundscape_unlabeled/shard-{0..15}.parquet` | 127,104 | 7.2 GB |
| Focal recordings | `focal/shard-{0..15}.parquet` | 265,924 | 15 GB |

### Partitioned data (created by PartitionFeatures task)

| Source | Path | Files | Purpose |
|--------|------|-------|---------|
| Focal by species | `focal/by_species/{species}.parquet` | 206 | Per-species processing |
| Unlabeled by site | `soundscape_unlabeled/by_site/{site}.parquet` | 21 | Per-site processing |

Use partitioned data for any per-entity analysis — avoids scanning all shards.

## Loading data

```python
import polars as pl
from birdclef_106.data_loading import (
    load_labeled, load_focal_species, load_site,
    list_focal_species, list_sites,
    logits_to_numpy, embeddings_to_numpy,
)

# Labeled soundscapes (739 rows)
df = load_labeled()

# Single species' focal data
df = load_focal_species("abethr1")
logits = logits_to_numpy(df)  # (N, 14795)

# Single site's unlabeled data
df = load_site("S01")

# List available species/sites
species = list_focal_species()  # ['abethr1', 'afrgrb1', ...]
sites = list_sites()            # ['S01', 'S02', ...]
```

## Label mapping

- `artifacts/perch_v2_labels.csv` — 14,796 scientific names (idx 0 = non-bird background)
- `artifacts/perch_v2_ebird_classes.csv` — eBird codes
- `artifacts/perch_species_mapping.json` — 203 direct BirdCLEF→Perch, 6 genus proxy, 25 unmapped
- Source: Kaggle model `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`

## Running analysis

```bash
cd user/anthony/experiments/106-perch-bayesian

# Run all phase 2 tasks (includes partition, prototypes, ridge, etc.)
sbatch sbatch/analyze.sbatch --phase 2

# Run all phases
sbatch sbatch/analyze.sbatch
```

Venv: `~/scratch/birdclef-2026/.venv-106`

## Analysis Luigi DAG

### Phase 1 (static baselines)
- `ZeroShotBaseline` — direct Perch logit readout for 203 mapped species
- `LogitCovariance` — 14795×14795 covariance from unlabeled soundscapes
- `EmbeddingCovariance` — 1536×1536 embedding covariance
- `DistributionShift` — focal vs soundscape feature distributions
- `PrototypeClassifier` — nearest-centroid in embedding space
- `FeatureSelection` — which Perch logits predict each BirdCLEF species
- `FocalSpeciesProfiles` — per-species logit signatures from focal data
- `SoundscapeProfiles` — per-site/diel activation patterns

### Phase 2 (models + structure)
- `EmpiricalBayesRidge` — ridge regression on embeddings (0.749 AUROC)
- `SparseLogistic` — sparse logistic on selected features
- `TemporalPooling` — per-species optimal window aggregation
- `SiteResidualProfiles` — James-Stein shrunk site profiles
- `SpeciesGraph` — species co-occurrence via window-level scoring
- `PartitionFeatures` — re-partition shards by species/site (one-time)
- `MultiPrototypeProfiles` — Leiden clustering on within-species residuals
- `SitePrototypes` — Leiden clustering on within-site residuals

### Key concepts
- **Star schema**: see `docs/star-schema.md`. Perch logit space (P=14795) is the
  shared dimension. All compositions (species×site, species×window) are matmuls
  through P. Centering + Mahalanobis weighting are critical.
- **Multi-prototype profiles**: within-species/site residual covariance → SVD
  whitening → dense Mahalanobis Gram matrix → Leiden community detection →
  variable-count prototypes per entity. Replaces single-mean profiles.
- **Cohesion filtering**: communities must exceed MIN_COHESION (mean pairwise
  cosine) to become prototypes. Rejects incoherent grab-bag clusters.

## Known gotchas

- Polars `read_csv` needs `infer_schema_length=10000` for train.csv (mixed
  numeric/string primary_label column)
- Console entry point `birdclef-106` has PATH issues on some compute nodes —
  sbatch scripts use `python -c "from birdclef_106.cli import main; main()"`
- Focal extraction needs 48 GB RAM (4 CPUs × 12 GB) — OOMs at 16 GB
- `_setup-venv.sh` uses `$HOME/arc/...` not `realpath` for cross-node stability
- Dense Gram matrix for large sites (S22: 40K windows) needs ~12 GB RAM

## Directory structure

```
├── proposal.md
├── tasks.md
├── CLAUDE.md                          # This file
├── pyproject.toml
├── configs/exp106.yaml
├── docs/
│   ├── analysis-ideas.md             # Full analysis plan with references
│   └── star-schema.md               # Dimensional analysis of Perch logit compositions
├── artifacts/
│   ├── perch_v2_labels.csv           # Perch taxonomy (14,796 names)
│   ├── perch_v2_ebird_classes.csv    # eBird codes
│   └── perch_species_mapping.json    # BirdCLEF → Perch index mapping
├── sbatch/
│   ├── _setup-venv.sh
│   ├── analyze.sbatch                # Run Luigi analysis pipeline
│   ├── extract-labeled.sbatch
│   ├── extract-unlabeled.sbatch
│   └── extract-focal.sbatch
├── src/birdclef_106/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── extract.py                    # Polars-based extraction pipeline
│   ├── perch_onnx.py                 # Perch v2 ONNX session wrapper
│   ├── audio.py                      # 32kHz audio loading + windowing
│   ├── data_loading.py               # Shared data loading (partitioned + sharded)
│   └── analysis/
│       ├── base.py                   # AnalysisTask base class
│       ├── partition.py              # Re-partition shards by species/site
│       ├── multi_prototype.py        # Multi-prototype Leiden clustering (species)
│       ├── site_prototypes.py        # Site-state Leiden clustering
│       ├── focal_profiles.py         # Per-species logit signatures
│       ├── species_graph.py          # Species co-occurrence graph
│       ├── site_residuals.py         # James-Stein shrunk site profiles
│       ├── ridge.py                  # Empirical Bayes ridge regression
│       └── ...                       # Other analysis modules
└── logs/                             # SLURM output logs
```
