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

**Phase A complete**: architecture + smoke tests passing.
**Phase B complete**: data pipeline + training loop + evaluation + sweep
driver.
**v2 cross-attention**: always-on. Species prototypes are
site-conditioned per window via multi-head cross-attention; `M_site`
feeds as K/V. The v1 scalar gate (proven hurtful at −0.045 AUROC) has
been removed.
**v2 auxiliary unsup loss**: the 127K unlabeled SC windows now shape
cross-attention via `unsupervised_cross_attn_energy`, in addition to
the direct `unsupervised_site_energy` on `M_site`. This closes the
v1-to-v2 transition gap (cross-attn was trained by only 262 labeled
windows per epoch).

**Results history on S22 decision fold:**

| Config | S22 AUROC | Δ vs ridge (0.749) |
|---|---|---|
| Ridge baseline (exp 106) | 0.749 | — |
| v1 no-gate (ReZero α=0) | **0.716** | −0.033 |
| v1 GPU (with gate) | 0.671 | −0.078 |
| v2 zero-init out_proj, labeled-only | 0.618 | −0.131 |
| v2 + softplus unsup aux | 0.5921 | −0.157 |
| **v3 contrastive (3ep, job 6487469)** | **0.6958** | **−0.053** |

**v3 status (2026-04-13)**: contrastive losses fix both v2 failure modes —
`L_site` is non-zero throughout (saturation bug gone), `L_xattn` decreases
without collapsing. **Val AUROC rose monotonically** through all 3 epochs
(0.617 → 0.685 → 0.696), opposite of v2's "peak early, regress" pattern.

But peak (0.6958) is still below v1-no-gate (0.716) — the model ran out
of LR (cosine decayed to 2.49e-08 by step 3100). Per-epoch gains were
0.069 → 0.011 → ? — diminishing but not flat. Most likely interpretation:
**3 epochs insufficient given the new contrastive losses; need 6+ epochs
with the cosine extended** to actually saturate.

Tier 2 dynamics check (all passed):
- `L_site` non-zero throughout (saturation bug fixed)
- `L_xattn` decreases without collapsing (no >10× drop)
- `unsup/lab` ratio stays in target [0.5, 2.0]
- Val AUROC rises with unsup loss decreasing (success signature)

Next likely move: 6-epoch run with extended cosine, OR try doubling
`lambda_xattn` since the xattn ratio is on the lower end.

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

Venv: `~/scratch/birdclef-2026/.venv-107` (dedicated, CUDA torch). Created
fresh by `sbatch/_setup-venv.sh` on first use.

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
│   ├── model.py                      # 5 nn.Modules: PerchProjection, 2 heads, cross-attn, top-level
│   ├── losses.py                     # 4-component compute_loss
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
│   ├── test.sbatch                   # All tests (model/data/gradient/cross-attn)
│   ├── train_loso.sbatch             # single-job LOSO
│   └── sweep_lambda_fam.sbatch       # single-job sweep
├── tests/                            # 7 files covering model + data + gradient invariants
└── logs/                             # SLURM output
```

## Architectural notes (v2 committed)

- **Scalar gate removed.** Replaced with species-site cross-attention.
  The v1 `g · sp + (1 − g) · sp · site_score` construction is gone.
- **Cross-attention is always on.** No feature flag. Focal recordings
  (`site_idx=None`) still use the shared species bank directly — this
  doubles as the OOD-site Kaggle fallback.
- **Zero-init `out_proj`.** Cross-attention is numerically identity at
  step 0. Gradient flows into `out_proj` from step 1 (unlocked at step
  2+ for the rest of the attention weights and `M_site` via the
  cross-attn path).
- **Two unsupervised terms.** `L_site` on `M_site` directly (v1, kept
  bounded via softplus); `L_site_xattn` on cross-attention-refined
  species retrieval (v2 new). The v2 term routes unlabeled SC gradient
  to `cross_attn.*` and `M_site`.
- **Detach policy on `L_site_xattn`.** Explicitly detaches `M_species`,
  `species.W_K.weight`, `species.log_β` — preserves v1's invariant that
  the species retrieval operator stays BCE-shaped.

## Known issues / gotchas

- **Data loading takes ~9 min per job.** Always prefer single jobs that
  load once. Avoid array jobs.
- **Default model has `n_sites=23`.** Matches the 23-site union, not the
  21 that appear in `by_site/` parquets.
- **Site energy fix is not in proposal.md**. Any future v2 rewrite of the
  proposal should document `softplus`-bounded site loss.
- **numpy string types** — `np.str_` from array indexing isn't the same
  as `str`; cast with `str(s)` before dict lookups in `site_to_idx`.
