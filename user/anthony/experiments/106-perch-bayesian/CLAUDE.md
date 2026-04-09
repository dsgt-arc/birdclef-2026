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
- `TemporalPooling` — per-species optimal window aggregation
- `SiteResidualProfiles` — James-Stein shrunk site profiles
- `SpeciesGraph` — species co-occurrence via window-level scoring
- `PartitionFeatures` — re-partition shards by species/site (one-time)
- `MultiPrototypeProfiles` — NMF (logits) + k-means (embeddings) per species
- `SitePrototypes` — NMF + k-means per site, diel characterization
- `PrototypeRidge` — ridge on prototype similarity features (MVP test)

### Key concepts
- **Star schema**: see `docs/star-schema.md`. Perch logit space (P=14795) is the
  shared dimension. All compositions (species×site, species×window) are matmuls
  through P. Centering is critical (cosine → Pearson correlation).
- **Multi-prototype profiles**: NMF on non-negative logits (parts-based, additive),
  k-means on signed embeddings. Fixed K per entity → clean (S,K,P) tensor.
- **Domain transfer**: 96.8% of logit dims shift focal↔soundscape. Transfer
  precision = decorrelation × domain reliability × species specificity.
- **Parallel spaces**: same pipeline on P (logits), E (embeddings), F (finetuned).
  Bayesian model averaging with species-dependent mixing weights.
- **Concept doc**: `docs/concepts/star-schema-tensor-decomposition.md` — tensor
  CP/NMF, TRMF temporal dynamics, Hopfield retrieval, Bayesian inference.

## Known gotchas

- Polars `read_csv` needs `infer_schema_length=10000` for train.csv (mixed
  numeric/string primary_label column)
- Console entry point `birdclef-106` has PATH issues on some compute nodes —
  sbatch scripts use `python -c "from birdclef_106.cli import main; main()"`
- Focal extraction needs 48 GB RAM (4 CPUs × 12 GB) — OOMs at 16 GB
- `_setup-venv.sh` uses `$HOME/arc/...` not `realpath` for cross-node stability
- NMF on 40K×14795 (site S22) takes ~5 min, needs ~5 GB RAM

## Directory structure

```
├── proposal.md
├── tasks.md
├── CLAUDE.md                          # This file
├── pyproject.toml
├── configs/exp106.yaml
├── docs/
│   ├── analysis-ideas.md             # Full analysis plan with references
│   ├── star-schema.md               # Star schema: compositions, prototypes, domain transfer
│   └── submission-strategy.md       # Bayesian ensemble calibration from LB submissions
├── artifacts/                        # Analysis outputs (JSON metrics, NPZ arrays)
├── sbatch/
│   ├── _setup-venv.sh
│   ├── analyze.sbatch                # Run Luigi analysis pipeline
│   ├── extract-labeled.sbatch
│   ├── extract-unlabeled.sbatch
│   └── extract-focal.sbatch
├── src/birdclef_106/
│   ├── cli.py
│   ├── extract.py                    # Polars-based extraction pipeline
│   ├── perch_onnx.py                 # Perch v2 ONNX session wrapper
│   ├── audio.py                      # 32kHz audio loading + windowing
│   ├── data_loading.py               # Shared data loading (partitioned + sharded)
│   └── analysis/
│       ├── base.py                   # AnalysisTask base class
│       ├── eda/                      # Exploratory / static baselines
│       │   ├── baseline_zero_shot.py # Perch logit readout (0.701 AUROC)
│       │   ├── covariance_logit.py   # 14795×14795 logit covariance
│       │   ├── covariance_embedding.py # 1536×1536 embedding covariance
│       │   ├── domain_shift.py       # Focal↔soundscape KS statistics
│       │   └── feature_importance.py # Per-logit AUROC per species
│       ├── structure/                # Star schema profiles, graphs, prototypes
│       │   ├── partition.py          # Shard → per-entity parquets
│       │   ├── profiles_focal.py     # Per-species logit signatures
│       │   ├── profiles_site.py      # James-Stein site profiles
│       │   ├── graph_species.py      # Species co-occurrence graph
│       │   ├── prototypes_species.py # NMF (logits) + k-means (embeddings) per species
│       │   └── prototypes_site.py    # NMF + k-means per site, diel characterization
│       ├── models/                   # Learned scoring models
│       │   ├── ridge_empirical_bayes.py  # Empirical Bayes ridge (0.749 AUROC)
│       │   ├── ridge_prototype.py    # Ridge on prototype similarity features
│       │   └── pooling_temporal.py   # Window aggregation strategies
│       └── _archived/               # Broken/superseded (not imported)
└── logs/                             # SLURM output logs
```
