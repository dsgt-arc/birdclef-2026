# Exp107 fast dev proxy benchmark

## Objective
Build a faster development split for experiment 107 that still adheres to the thesis: all three streams remain present (strong labeled soundscapes, weak focal recordings, unlabeled soundscapes), site-conditioning is still exercised, and the validation set is less misleading than S22-only. The goal is not just raw speed; it is a quick-turn proxy that preserves the hard parts of the problem, especially bird vs amphibian and with-focal vs without-focal behavior.

## Metrics
- **Primary**: `dev_auroc` (higher is better) — macro AUROC on the fast development validation split
- **Secondary**: `benchmark_seconds`, `without_focal_auroc`, `with_focal_auroc`, `bird_auroc`, `amphibian_auroc`, `best_step`, `n_val_windows` — monitor whether the split remains representative and whether iteration speed is actually improving

## How to Run
`bash scripts/benchmark_exp107_dev.sh` — activates the existing scratch venv directly inside the current GPU allocation, runs the fast-split training/eval command, then emits `METRIC ...` lines from the saved JSON results.

Useful overrides:
- `DEV_SITES=S13,S15,S19 bash scripts/benchmark_exp107_dev.sh`
- `FOCAL_CACHE_SIZE=16384 UNSUP_CACHE_SIZE=16384 bash scripts/benchmark_exp107_dev.sh`
- `ATTN_DROPOUT=0.05 bash scripts/benchmark_exp107_dev.sh`
- `STRONG_STRIDE=2 bash scripts/benchmark_exp107_dev.sh`
- `LABEL_SMOOTHING=0.0 bash scripts/benchmark_exp107_dev.sh`
- `CKPT_SUBDIR=exp107-manual-test bash scripts/benchmark_exp107_dev.sh`

The first run builds a local NPZ cache under `${TMPDIR:-/tmp}/birdclef-107-devcache/<config-key>/`; later runs reuse the matching cache. Different focal/unsup cache sizes now get separate keyed cache directories, so switching configs does not destroy fast reuse for earlier settings.

## Files in Scope
- `src/birdclef_107/model.py` — cross-attention and prototype retrieval behavior
- `src/birdclef_107/losses.py` — labeled / unsupervised / family loss composition
- `scripts/train.py` — training schedule, warmup, validation cadence, early stopping
- `src/birdclef_107/eval.py` — AUROC and subset diagnostics
- `sbatch/train_loso.sbatch` — runtime hyperparameter plumbing if needed
- `tests/` — update/add model and gradient-routing tests when behavior changes

## Off Limits
- Feature parquet caches under `~/scratch/birdclef-2026/features/exp106`
- Experiment 106 source tree
- Large workflow changes that require array jobs or repeated data reloads
- New heavyweight dependencies

## Constraints
- Preserve the v3/v4 gradient-routing invariants documented in `design.md` / `model.py`
- Run directly inside the current GPU `salloc`; do **not** launch `sbatch` from autoresearch
- Do not create `.venv` in the repo or home dir; activate `~/scratch/birdclef-2026/.venv-107`
- Prefer split/data-loader changes that preserve the thesis rather than turning the task into a toy subset
- Passing model/cross-attention/gradient tests required before keeping code changes

## What's Been Tried
- **Old proxy was too narrow**: S22-only is easy to iterate on conceptually but is dominated by amphibian-heavy and with-focal-heavy positives; it is a poor quick proxy for the `without_focal` failure mode.
- **Current candidate fast dev holdout** from labeled-only analysis: `S13,S15,S19` gives 108 validation windows with 36 evaluable species, covering birds, amphibians, and 14 without-focal species — a better balance than S22 alone without needing full LOSO.
- **Direct venv activation is available**: the scratch env at `~/scratch/birdclef-2026/.venv-107` works inside the GPU allocation, so benchmark overhead should avoid `sbatch` and reuse the allocated node.
- **Implemented speed lever**: use a local `/tmp` NPZ cache for a subsampled focal + unlabeled training subset while keeping the full labeled soundscape set. This preserves the three-stream thesis but removes repeated parquet scans on every run.
- **Avoid**: reverting to v2 softplus unsupervised losses; they saturated/collapsed and are explicitly ruled out by prior experiments.
