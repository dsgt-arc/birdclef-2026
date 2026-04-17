# exp107 training-data cache

Persistent, sharded cache that backs `scripts/train.py` and
`bash scripts/benchmark_exp107_dev.sh`. Replaces the old per-`(focal_size, unsup_size,
seed)` monolithic NPZ bundle under `$TMPDIR` with a split base + index
layout on scratch.

## Why a cache

The exp106 parquet feature store is ~45 GB on disk, mostly because
every row carries a 14795-dim `perch_logits` column that exp107 does
not use. A cold parquet scan was ~933 s; warm same-config reuse was
~81 s. Any change to `FOCAL_CACHE_SIZE`, `UNSUP_CACHE_SIZE`, or `SEED`
re-paid the full cold cost. The new cache stores stripped NPZ shards
on Lustre scratch so that:

- **Cold build** happens once per machine-state (not per SLURM job).
- **Size/seed sweeps** only regenerate kilobyte-scale index files.
- **Compact Perch sidecars remain reachable** as an opt-in cache path for
  iterations that want teacher features without storing all 14,795 logits.

## Layout

```
$HOME/scratch/birdclef-2026/cache/exp107/
├── base-v1/
│   ├── base_manifest.json
│   ├── labeled.npz                  # 739 rows, all fields, ~5 MB
│   ├── labeled_perch.npz            # OPTIONAL, ~44 MB
│   ├── focal/shard-NN.npz           # 16 files × ~100 MB
│   ├── focal_perch/shard-NN.npz     # OPTIONAL, 16 files × ~980 MB
│   ├── unsup/shard-NN.npz           # 16 files × ~50 MB
│   └── unsup_perch/shard-NN.npz     # OPTIONAL, 16 files × ~470 MB
└── idx-v1/
    ├── focal-n{N}-seed{S}.npy       # ~KB each
    └── unsup-n{N}-seed{S}.npy
```

- **Default size on disk**: ~2.4 GB.
- **With compact Perch sidecars**: modestly larger than core-only. Each
  row stores dense mapped 234-d probabilities plus a sparse top-64
  residual teacher channel instead of all 14,795 logits.

Row order inside `*_perch/shard-NN.npz` is identical to
`*/shard-NN.npz`, so the same global indices in `idx-v1/` address
both.

## Using it

### Recommended: packaged CLI + coordinated SLURM pipeline

This is the recommended path for cold builds and for scaling to larger
feature stores. The build logic now lives in the package CLI under
`birdclef-107 cache ...`, while a single coordinator sbatch job handles
parallelization by running bootstrap/finalize/verify locally and
fanning shard workers out on the same node.

Main entrypoint:

```bash
# Default: coordinator job, shard workers capped at 4 concurrent tasks
sbatch sbatch/cache_pipeline.sbatch

# Also build compact Perch sidecars
INCLUDE_PERCH_FEATURES=1 sbatch sbatch/cache_pipeline.sbatch

# Refresh the common dev subset indexes and run final verification
INCLUDE_PERCH_FEATURES=1 FOCAL_CACHE_SIZE=32768 UNSUP_CACHE_SIZE=32768 \
  CACHE_SUBSET_SEED=42 VERIFY_CACHE=1 sbatch sbatch/cache_pipeline.sbatch
```

The coordinator internally runs:

1. `birdclef-107 cache bootstrap`
2. `birdclef-107 cache shard` over all work units with bounded same-node fanout
3. `birdclef-107 cache finalize`
4. `birdclef-107 cache verify` (optional, on by default in the coordinator)

Relevant overrides:

| Env var | Default | Effect |
|---|---|---|
| `FEATURES_DIR` | `$HOME/scratch/birdclef-2026/features/exp106` | Source parquet root. |
| `CACHE_ROOT` | `$HOME/scratch/birdclef-2026/cache/exp107` | Cache root containing `base-v1/` and `idx-v1/`. |
| `PARALLEL_WORKERS` | `4` | Max same-node shard workers. Must stay `<16` and `<= cpus-per-task`; this is the main scale knob. |
| `INCLUDE_PERCH_FEATURES` | `0` | Set to `1` to build the optional compact `*_perch/*.npz` sidecars (mapped 234-d probs + sparse top-64 residual activations). `INCLUDE_PERCH_LOGITS` remains an alias. |
| `CACHE_FORCE_REBUILD_BASE` | `0` | Set to `1` to rebuild metadata + shard outputs from parquet. |
| `CACHE_FORCE_REBUILD` | `0` | Set to `1` to force the optional subset index files to rebuild in stage 3. |
| `FOCAL_CACHE_SIZE` | `0` | If >0, build `idx-v1/focal-n{N}-seed{S}.npy` during stage 3. |
| `UNSUP_CACHE_SIZE` | `0` | If >0, build `idx-v1/unsup-n{N}-seed{S}.npy` during stage 3. |
| `CACHE_SUBSET_SEED` | `42` | Seed for the optional subset index files. |

Why this scales better:

- **Single packaged CLI.** Cache logic lives in `src/birdclef_107/cli.py`
  and `cache_pipeline.py`, not in one-off script wrappers.
- **One coordinator job.** `sbatch/cache_pipeline.sbatch` owns the orchestration.
- **Dynamic work count.** The coordinator asks `birdclef-107 cache plan --count-only`.
- **Bounded concurrency.** `PARALLEL_WORKERS` keeps same-node shard fanout under control.
- **Still idempotent.** Stage 2 workers skip fresh outputs by comparing
  output mtimes against source parquet mtimes and by rejecting stale
  pre-compact sidecar files.


### From `birdclef-107` directly

The packaged CLI exposes the cache stages directly:

```bash
birdclef-107 cache plan --count-only
birdclef-107 cache bootstrap --features-dir "$FEATURES_DIR" --cache-dir "$CACHE_ROOT"
birdclef-107 cache shard --features-dir "$FEATURES_DIR" --cache-dir "$CACHE_ROOT" --task-id 0
birdclef-107 cache finalize --features-dir "$FEATURES_DIR" --cache-dir "$CACHE_ROOT"
birdclef-107 cache verify --features-dir "$FEATURES_DIR" --cache-dir "$CACHE_ROOT"
```

This is what `sbatch/cache_pipeline.sbatch` calls under the hood.

### From `scripts/benchmark_exp107_dev.sh`

Default configuration needs nothing extra. The script passes
`--cache-dir "$CACHE_ROOT"` with
`CACHE_ROOT=$HOME/scratch/birdclef-2026/cache/exp107`.

Relevant overrides:

| Env var | Default | Effect |
|---|---|---|
| `CACHE_ROOT` | `$HOME/scratch/birdclef-2026/cache/exp107` | Root directory containing `base-v1/` and `idx-v1/`. Point at another scratch location if you want a sandbox. |
| `FOCAL_CACHE_SIZE` | `32768` | Rows in the focal subsample (indexed into the full base shards). `0` = full corpus. |
| `UNSUP_CACHE_SIZE` | `32768` | Rows in the unlabeled subsample. `0` = full corpus. |
| `SEED` | `42` | Seed for `_weighted_focal_subset` / `_stratified_unsup_subset`. |
| `INCLUDE_PERCH_LOGITS` | unset | Set to `1` to hydrate the optional compact Perch sidecar fields on all three dataclasses. Triggers sidecar build on first use. |

Example: sweep focal subset size while keeping everything else:

```bash
FOCAL_CACHE_SIZE=16384 bash scripts/benchmark_exp107_dev.sh
FOCAL_CACHE_SIZE=65536 bash scripts/benchmark_exp107_dev.sh
```

Both runs read the same `base-v1/` shards and reuse the same
`unsup-n32768-seed42.npy`; only the focal index file is new per call.

### From `scripts/train.py` directly

```bash
python scripts/train.py \
  --cache-dir "$HOME/scratch/birdclef-2026/cache/exp107" \
  --focal-cache-size 32768 \
  --unsup-cache-size 32768 \
  --seed 42 \
  [other training flags...]
```

Relevant cache flags:

| Flag | Purpose |
|---|---|
| `--cache-dir PATH` | Cache root. Omit entirely to bypass the cache and read parquet every time. |
| `--focal-cache-size N` | Rows in focal subset; `0` means "use every focal row". |
| `--unsup-cache-size N` | Same for unlabeled soundscapes. |
| `--seed N` | Seed for subsample selection (separate from training seed). |
| `--include-perch-logits` | Populate `*.perch_mapped`, `*.perch_topk_idx`, and `*.perch_topk_val`. Builds sidecars on demand. |
| `--cache-force-rebuild` | Re-derive the `idx-v1/` files even if they exist. Leaves `base-v1/` alone. |
| `--cache-force-rebuild-base` | Re-derive the full `base-v1/` from parquet. Use if you suspect cache corruption or if a source parquet changed without its mtime advancing. |

### From Python

```python
from pathlib import Path
from birdclef_107.cache import load_training_data_cached

sc, focal, unsup = load_training_data_cached(
    features_dir=str(Path.home() / "scratch/birdclef-2026/features/exp106"),
    cache_root=Path.home() / "scratch/birdclef-2026/cache/exp107",
    focal_size=32768,
    unsup_size=32768,
    seed=42,
    include_perch_logits=False,
)
```

Returns the same three dataclasses the rest of exp107 already uses.
With `include_perch_logits=True`, each dataclass gets three additional
row-aligned fields:

- `perch_mapped`: `(n_subset, 234)` float32 mapped BirdCLEF probabilities
- `perch_topk_idx`: `(n_subset, 64)` int32 raw teacher class ids
- `perch_topk_val`: `(n_subset, 64)` float32 residual teacher probabilities

## Invalidation

The loader decides whether a cached shard is still good by comparing
its manifest entry against the source parquet's `stat().st_mtime_ns`.

- **One shard's parquet changed** → only that shard's NPZ is rebuilt;
  the other 15 keep their inode and mtime.
- **A size or seed changed** → `base-v1/` is untouched; only the
  affected `idx-v1/*.npy` file is recomputed (tens of milliseconds on
  the labels/site arrays alone).
- **Perch sidecar features requested for the first time** → core shards
  are left alone; `focal_perch/`, `unsup_perch/`, `labeled_perch.npz`
  are built on demand from parquet.
- **You think something went wrong** → delete `base-v1/` manually, or
  run with `--cache-force-rebuild-base`. The next call pays the full
  ~10–15 min cold cost.

Never edit `base_manifest.json` by hand. If you think the mtime
fingerprint is wrong, force-rebuild; that's cheaper than debugging a
stale manifest.

## Concurrency

Two SLURM jobs can cold-build simultaneously without corrupting the
cache:

- Per-shard writes go to `focal/shard-NN.partial-{pid}.npz`, then
  `os.replace` to the final name. Readers never see a partial file.
- `base_manifest.json` is protected by `fcntl.flock` on
  `<cache_root>/.manifest.lock` during the read-merge-write cycle.
- Index files `np.save` to a `.partial` path then `os.replace`.
  Identical inputs produce identical bytes, so races are harmless.

If a job crashes mid-build, you may see `.partial-{pid}.npz` files
left behind. They're ignored by the loader and safe to delete
(`find base-v1 -name '*.partial-*' -delete`).

## Reclaiming scratch

```bash
# Drop only the perch sidecars (keeps the fast path)
rm -rf $HOME/scratch/birdclef-2026/cache/exp107/base-v1/{focal_perch,unsup_perch}
rm -f  $HOME/scratch/birdclef-2026/cache/exp107/base-v1/labeled_perch.npz

# Drop index files (force a per-config re-sample)
rm -rf $HOME/scratch/birdclef-2026/cache/exp107/idx-v1

# Nuclear: wipe everything, next run pays the full cold cost
rm -rf $HOME/scratch/birdclef-2026/cache/exp107
```

## Expected wall-clock

| State | Expected load time |
|---|---|
| Cold build, no perch | ~10–15 min (parquet scans dominate) |
| Cold build, with perch sidecars | ~20–30 min (still reads the raw `perch_logits` parquet column, then compresses to mapped+sparse sidecars) |
| Warm reuse, same `(N, seed)` | <10 s |
| Warm reuse, different `focal_size` only | <10 s (index rebuild is <1 s; materialization dominates) |
| Warm reuse, different seed only | <10 s |
| Shard-level rebuild (one parquet touched) | ~1–2 min |
