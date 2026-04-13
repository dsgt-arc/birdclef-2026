# Experiment 107: Hopfield Prototype Retrieval

## What this experiment does

Trains a factored star-schema Hopfield retrieval model (species × site
prototype banks) over a learned projection of frozen Perch v2 embeddings.
Uses three data streams through one retrieval mechanism:

1. **Strong SC** (739 labeled soundscape windows) — anchor to target domain
2. **Weak focal** (~265K bag-labeled focal recordings) — shape species bank
3. **Unlabeled SC** (127K soundscape windows) — shape site bank via Hopfield energy

Four-term loss (v3 as-built): `L_lab + λ_site · L_site + λ_xattn · L_xattn + λ_fam · L_fam`.
Family hierarchy is a training-time shrinkage regularizer (no inference
cost). See `proposal.md` and `design.md`.

## Status (2026-04-13)

**v4 smoke on S22 beats exp 106 ridge — but only on S22, only on one
seed, and still 18 AUROC points below the Kaggle LB top (~0.94).**
Do not interpret S22 success as LB competitiveness.

**Stack**: v2 cross-attention (scalar gate removed) + v3 InfoNCE
contrastive losses (softplus bounded losses removed) + v4 regularization
(dropout 0.1 × 2, wd 1e-2, label smoothing 0.01, strong_stride 4).

**Results history on S22 decision fold:**

| Config | S22 AUROC | Δ vs ridge (0.749) |
|---|---|---|
| Ridge baseline (exp 106) | 0.749 | — |
| v1 no-gate (ReZero α=0) | 0.716 | −0.033 |
| v1 GPU (with gate) | 0.671 | −0.078 |
| v2 zero-init out_proj, labeled-only | 0.618 | −0.131 |
| v2 + softplus unsup aux | 0.5921 | −0.157 |
| v3 contrastive 3-ep | 0.6958 | −0.053 |
| v3 contrastive 20-ep | 0.7074 | −0.042 |
| **v4 regularized (job 6504860)** | **0.7613** | **+0.012** |

**v4 details (2026-04-13)**: peak at step 600 (58% of epoch 0,
immediately after unsup warmup), early-stopped at step 1600.
L_lab drop rate 1.4× across 1.6 epochs (vs v3's 10× across 3
epochs) → regularization is working. 6 min training on RTX 6000.

**v4 per-subset breakdown (the load-bearing part)**:

| Subset | n | v3 | **v4** | Δ |
|---|---|---|---|---|
| bird | 162 | 0.7264 | **0.8683** | +0.142 |
| amphibian | 35 | 0.6986 | 0.6529 | −0.046 |
| insect | 28 | 0.0 | 0.0 | — (no positives in fold) |
| with_focal | 206 | 0.7015 | 0.7688 | +0.067 |
| **without_focal** | **28** | **0.7874** | **0.6607** | **−0.127** |

v3 had `without_focal > with_focal`; v4 flipped it. Regularization
helped 206 with-focal species (bird-heavy) by +0.067 but hurt the 28
without-focal species by −0.127. Birds dominate 69% of the label space,
so the bird win carries the average — **but if the hidden LB test set
is less bird-heavy than focal suggests, v4's gain may invert**.

## Critical caveats (read before celebrating)

1. **LB gap is 0.18.** Top BC2026 is ~0.94. "Beats ridge on S22" is a
   within-domain comparison on 477 windows of the largest labeled
   site. It does **not** imply LB competitiveness. Missing pieces we
   don't have: pseudo-labeling, SED/attention temporal head, full
   Perch fine-tuning, ensembling, audio-level augmentation, TTA.
2. **Single fold, single seed.** S22 is 65% of labeled SC windows
   (477/739). Other 8 folds not yet measured at v4. 0.012 Δ may be
   within seed-to-seed variance. Full LOSO + seed replicates are
   required to call v4 a real lift.
3. **`without_focal` regression** (above). 28 species lost 0.127 AUROC
   — the exact subset the cross-attention architecture was supposed
   to help (they have no focal prior to lean on). v4 regularization
   may be miscalibrated for this subset; a v4.1 with `attn_dropout=0.05`
   is a natural diagnostic.
4. **Peak at 2.9% of planned training.** 600 / 20760 steps. Either
   the model converges in <1 epoch, or the regularization budget is
   blocking further improvement. Not distinguished.

## Next-action decision tree

1. **Primary next step**: full 9-fold LOSO (`FOLD=all sbatch sbatch/train_loso.sbatch`).
   Single job, ~1h for data + ~1h for training. Confirms or disconfirms
   the S22 +0.012 as a cross-fold phenomenon. Unblocks the pre-registered
   `λ_fam` sweep.
2. If LOSO clears pooled-AUROC 0.72 → run the `λ_fam` sweep as the
   pre-registered ablation, then prepare a Kaggle submission.
3. If LOSO lands 0.70–0.72 → consider Bin B (Perch pseudo-labels on
   unlabeled SC, targets the bird subset orthogonally via projection +
   species head, uses exp 106's `perch_species_mapping.json`).
4. If LOSO lands <0.70 or the `without_focal` regression dominates on
   other folds → trigger Bin C fallback (ship v1-no-gate + ridge
   probability-avg ensemble).

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

### Unsup losses: contrastive, not softplus

v2 used `softplus(-agg).mean()` for `L_site` and
`softplus(-logsumexp_s per_species)` for `L_xattn`. Both failed: the
first saturated at init (zero gradient to `M_site`), the second
collapsed to "match anything" (corrupting labeled retrieval). v3
replaced both with **InfoNCE** formulations that stay gradient-rich
at saturation and force site-conditioning to be *meaningful*. At random
init both losses ≈ `log(23) ≈ 3.14` — no zero-gradient regime. Don't
revert to softplus.

### Training regime: outer loop on focal + strong stride

`steps_per_epoch = len(focal_loader)` (~1000), not `len(strong_loader)`
(2). But v3 showed that cycling strong every focal step → each of 262
strong windows seen ~519×/epoch = 10,380× in 20 epochs → labeled
overfit. v4 adds `strong_stride=4`: compute strong BCE only every 4th
step. Strong windows seen ~130×/epoch. Implemented via `include_strong`
kwarg to `compute_loss`. Default `epochs=20` gives cosine LR full range.
Intra-epoch validation every 100 steps catches the peak (v4 peaked at
step 600 of 20760).

### v4 regularization budget

`attn_dropout=0.1`, `proj_dropout=0.1`, `weight_decay=1e-2`,
`label_smoothing=0.01`, `strong_stride=4`. Rationale in `design.md`
§"Schedule". Without this v3 peaked at epoch 2 (0.7074, overfit); with
it v4 peaked at step 600 (0.7613, regularization binding).

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
│   ├── losses.py                     # 4-component compute_loss (lab + site + xattn + fam; v4 adds label_smoothing + include_strong kwargs)
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

## Architectural notes (v2/v3/v4 as-built)

- **Scalar gate removed (v2).** Replaced with species-site cross-attention.
  The v1 `g · sp + (1 − g) · sp · site_score` construction is gone.
- **Cross-attention is always on (v2).** No feature flag. Focal recordings
  (`site_idx=None`) still use the shared species bank directly — this
  doubles as the OOD-site Kaggle fallback.
- **Zero-init `out_proj` (v2).** Cross-attention is numerically identity
  at step 0. Gradient flows into `out_proj` from step 1 (unlocked at
  step 2+ for the rest of the attention weights).
- **Two contrastive unsup terms (v3).** `L_site_contrastive` (InfoNCE
  over 23 sites on direct `M_site` retrieval) + `L_xattn_contrastive`
  (site-InfoNCE on cross-attn-refined species profiles). Both replace
  v2's softplus-bounded forms.
- **Full detach policy inside `L_xattn_contrastive` (v3).** Explicitly
  detaches `M_species`, `species.W_K.weight`, `species.log_β`, AND
  `M_site`. Clean one-objective-per-parameter: `L_site` exclusively
  shapes `M_site`, `L_xattn` exclusively shapes `cross_attn.*`.
- **Regularization budget (v4).** `attn_dropout=0.1`, `proj_dropout=0.1`,
  `weight_decay=1e-2`, `label_smoothing=0.01`, `strong_stride=4`. Fixes
  v3's labeled overfit (peak at ep 2 → v4 peak at step 600 with a
  +0.054 AUROC gain on S22).

## Known issues / gotchas

- **Data loading takes ~9 min per job.** Always prefer single jobs that
  load once. Avoid array jobs.
- **Default model has `n_sites=23`.** Matches the 23-site union, not the
  21 that appear in `by_site/` parquets.
- **v4 `without_focal` regression on S22.** 28 species dropped 0.127
  AUROC vs v3. Regularization budget may be too aggressive for the
  cross-attn-only path. A v4.1 with `attn_dropout=0.05` is the obvious
  ablation.
- **LB gap is 0.18.** v4 S22 = 0.7613; LB top ≈ 0.94. This architecture
  has no pseudo-labeling, no SED, no full Perch fine-tuning, no
  ensemble, no audio-level aug, no TTA. Do not confuse "beats internal
  ridge on S22" with "competitive on LB".
- **Peak at step 600 of 20760 (2.9%).** Either the model converges
  early or regularization is too aggressive. Not distinguished.
- **numpy string types** — `np.str_` from array indexing isn't the same
  as `str`; cast with `str(s)` before dict lookups in `site_to_idx`.
