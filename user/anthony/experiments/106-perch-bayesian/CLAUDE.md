# Experiment 106: Perch v2 Bayesian Label Transfer

## What this experiment does

Extracts Perch v2 ONNX embedding (1536-d) and label logits (14,795-d) for all
BirdCLEF 2026 audio, then applies Bayesian/statistical methods to transfer
Perch's pretrained taxonomy to BirdCLEF 2026's 234 targets.

## Status

Phase 1 (extraction) COMPLETE. Phase 2 (analysis) NOT STARTED.

## Extracted data

All on scratch at `~/scratch/birdclef-2026/features/exp106/`:

| Source | Path | Rows | Size |
|--------|------|------|------|
| Labeled soundscapes | `soundscape_labeled.parquet` | 739 | 43 MB |
| Unlabeled soundscapes | `soundscape_unlabeled/shard-{0..15}.parquet` | 127,104 | 7.2 GB |
| Focal recordings | `focal/shard-{0..15}.parquet` | 265,924 | 15 GB |

Parquet schema (polars `List(Float32)` columns):
```
filename: String, window_idx: Int64, start_sec: Float64,
source: String, split: String, primary_label: String,
ground_truth: String, embedding: List(Float32) [1536],
perch_logits: List(Float32) [14795]
```

## Label mapping

- `artifacts/perch_v2_labels.csv` — 14,796 scientific names (idx 0 = non-bird background)
- `artifacts/perch_v2_ebird_classes.csv` — eBird codes
- `artifacts/perch_species_mapping.json` — 203 direct BirdCLEF→Perch, 6 genus proxy, 25 unmapped
- Source: Kaggle model `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`

## Running extraction

```bash
cd user/anthony/experiments/106-perch-bayesian

# Labeled soundscapes (single job, ~5 min)
sbatch sbatch/extract-labeled.sbatch

# Unlabeled soundscapes (16 shards, ~20 min/shard)
sbatch sbatch/extract-unlabeled.sbatch

# Focal recordings (16 shards, ~25 min/shard, needs 48 GB)
sbatch sbatch/extract-focal.sbatch
```

Venv: `~/scratch/birdclef-2026/.venv-106` (polars, onnxruntime, torch CPU).

## Loading data

```python
import polars as pl
import numpy as np

# Single file
df = pl.read_parquet("~/scratch/birdclef-2026/features/exp106/soundscape_labeled.parquet")

# All shards (lazy scan)
df = pl.scan_parquet("~/scratch/birdclef-2026/features/exp106/focal/shard-*.parquet")

# Extract numpy arrays from list columns
emb = np.stack(df["embedding"].to_list())       # (N, 1536)
logits = np.stack(df["perch_logits"].to_list())  # (N, 14795)
```

## Analysis plan

See `docs/analysis-ideas.md` for full plan with references. Summary:

**Phase 1 (static):** zero-shot logit readout → empirical Bayes ridge on embeddings
→ distribution shift diagnostics → prototype classifier for insect sonotypes

**Phase 2 (temporal):** simple temporal pooling → HMM diel regimes → DMD on
per-site time series

**Phase 3 (full pipeline):** UKF temporal smoothing → classification-occupancy
model → GP classification

## Known gotchas

- Polars `read_csv` needs `infer_schema_length=10000` for train.csv (mixed
  numeric/string primary_label column)
- Console entry point `birdclef-106` has PATH issues on some compute nodes —
  sbatch scripts use `python -c "from birdclef_106.cli import main; main()"`
- Focal extraction needs 48 GB RAM (4 CPUs × 12 GB) — OOMs at 16 GB
- `_setup-venv.sh` uses `$HOME/arc/...` not `realpath` for cross-node stability

## Directory structure

```
├── proposal.md
├── tasks.md
├── CLAUDE.md                          # This file
├── pyproject.toml
├── configs/exp106.yaml
├── docs/
│   └── analysis-ideas.md             # Full analysis plan with references
├── artifacts/
│   ├── perch_v2_labels.csv           # Perch taxonomy (14,796 names)
│   ├── perch_v2_ebird_classes.csv    # eBird codes
│   └── perch_species_mapping.json    # BirdCLEF → Perch index mapping
├── sbatch/
│   ├── _setup-venv.sh
│   ├── extract-labeled.sbatch
│   ├── extract-unlabeled.sbatch
│   └── extract-focal.sbatch
├── src/birdclef_106/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── extract.py                    # Polars-based extraction pipeline
│   ├── perch_onnx.py                 # Perch v2 ONNX session wrapper
│   └── audio.py                      # 32kHz audio loading + windowing
└── logs/                             # SLURM output logs
```
