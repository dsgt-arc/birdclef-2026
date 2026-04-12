# Experiment 107: Hopfield Prototype Retrieval

## What this experiment does

Trains a factored star-schema Hopfield retrieval model (species × site
prototype banks) over a learned projection of frozen Perch v2 embeddings.
Uses three data streams through one retrieval mechanism:

1. **Strong SC** (739 labeled soundscape windows) — anchor to target domain
2. **Weak focal** (~265K bag-labeled focal recordings) — shape species bank
3. **Unlabeled SC** (127K soundscape windows) — shape site bank via Hopfield energy

Three-term loss: `L_labeled + λ_site · L_site_unsup + λ_fam · L_family`.
Family hierarchy is a training-time shrinkage regularizer (no inference
cost). See `proposal.md` and `design.md`.

## Status (2026-04-12)

**Phase A complete**: architecture + 13 smoke tests passing.
**Phase B complete**: data pipeline + training loop + evaluation + sweep
driver. 26/26 combined tests passing.

**First real result (smoke test on S22, job 6463628)**: **0.658 AUROC**
— below the 0.749 ridge baseline from exp 106. Several debugging issues
resolved during smoke testing; see `../../..//.claude/projects/.../memory/exp-107-hopfield-prototypes.md`
for the debugging log.

**Current test (job 6467550)**: restructured training regime (outer loop
on focal, not strong). Tests whether the underperformance was a training
budget issue or a deeper architectural limit.

## Data dependency

Reuses the exp 106 parquet cache at `~/scratch/birdclef-2026/features/exp106/`:

| Source | Path | Rows | Used by |
|--------|------|------|---------|
| Labeled SC | `soundscape_labeled.parquet` | 739 | strong BCE |
| Unlabeled SC | `soundscape_unlabeled/by_site/*.parquet` | 127K | site Hopfield energy |
| Focal | `focal/shard-*.parquet` (all 16 shards) | 265K | weak BCE |

Data loading takes ~9 min per job (focal dominates). **Do not use array
jobs** — they reload the same data N times. Prefer a single job with
`--fold all` (sequential) for LOSO, and the sweep driver for lambda_fam.

## Key design decisions

### Site energy must be bounded

The proposal's pseudocode uses `L_site = -mean(lse/beta)` which is unbounded
below. In practice this diverges to -∞ and dominates the labeled loss. The
shipped code uses `softplus(-agg).mean()` (equivalent to BCE against all-ones),
which preserves the gradient direction but bounds the loss to `[0, ∞)` with
natural saturation near 0. **Don't revert this.**

### Training regime: outer loop on focal

`steps_per_epoch = len(focal_loader)` (~1000), not `len(strong_loader)`
(2). Otherwise focal is barely sampled while strong SC is seen ~50x per
epoch. Default `epochs=3` gives ~3000 total steps with balanced stream
coverage. Intra-epoch validation every 100 steps catches the peak.

### Site index must be UNION of labeled + unlabeled

Labeled SC has some sites (S08, S23) that don't appear in
`soundscape_unlabeled/by_site/`. `load_labeled_soundscape` builds
`site_to_idx` from the union — 23 sites total, S01–S23. Default model
`n_sites=23`.

### CUDA module load must be in the sbatch body

`module load cuda/12.6.1` goes in `sbatch/train_loso.sbatch` directly, NOT
in `sbatch/_setup-venv.sh`. When loaded inside a sourced script, it
silently fails and torch falls back to CPU without error.

## Running

```bash
# Smoke test (S22 decision fold, default)
sbatch sbatch/train_loso.sbatch

# Full 9-fold LOSO (single sequential job, loads data once)
FOLD=all sbatch sbatch/train_loso.sbatch

# Pre-registered lambda_fam ablation on S22 (5 values, data loaded once)
sbatch sbatch/sweep_lambda_fam.sbatch

# Tests
sbatch sbatch/test.sbatch
```

Venv: `~/scratch/birdclef-2026/.venv-106` (shared with exp 106).

## Directory structure

```
├── proposal.md                       # Thesis, design, evaluation plan
├── design.md                         # Interface spec & math
├── CLAUDE.md                         # This file
├── pyproject.toml                    # exp106-perch-bayesian as dep
├── docs/
│   ├── star-schema-dimensional-analysis.md
│   └── families.csv                  # species → family (iNat API)
├── src/birdclef_107/
│   ├── model.py                      # PerchProjection + 2 heads + composition
│   ├── losses.py                     # 3-term compute_loss
│   ├── family_utils.py               # build_family_index
│   ├── data.py                       # 3-stream loaders + LOSO splits
│   ├── eval.py                       # pooled AUROC + subsets
│   └── assets/families.csv           # bundled package asset
├── scripts/
│   ├── train.py                      # Training loop (outer=focal)
│   ├── sweep_lambda_fam.py           # 5-point ablation driver
│   ├── analyze_coverage.py
│   └── fetch_families.py
├── sbatch/
│   ├── _setup-venv.sh
│   ├── test.sbatch                   # 26 Phase A+B tests
│   ├── train_loso.sbatch             # single-job LOSO
│   └── sweep_lambda_fam.sbatch       # single-job sweep
├── tests/                            # 26 tests (13 Phase A + 13 Phase B)
└── logs/                             # SLURM output
```

## Open architectural question

Proposal v1's scalar site gate (`g * species + (1-g) * species * site_score`)
explicitly under-shares information between banks. The proposal itself
flags this as a v2 candidate: "low-rank site→species projection." If
the current training regime fix doesn't close the gap to ridge 0.749,
the natural next step is cross-bank attention (species prototypes attend
to site prototypes per window). The `affinity_matrix` function already
computes this for interpretation — promoting it to the forward pass is
the concrete v2 change.

## Known issues / gotchas

- **Data loading takes ~9 min per job.** Always prefer single jobs that
  load once. Avoid array jobs.
- **Default model has `n_sites=23`.** Matches the 23-site union, not the
  21 that appear in `by_site/` parquets.
- **Site energy fix is not in proposal.md**. Any future v2 rewrite of the
  proposal should document `softplus`-bounded site loss.
- **numpy string types** — `np.str_` from array indexing isn't the same
  as `str`; cast with `str(s)` before dict lookups in `site_to_idx`.
