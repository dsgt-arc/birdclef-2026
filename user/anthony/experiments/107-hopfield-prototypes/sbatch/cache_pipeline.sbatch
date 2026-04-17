#!/bin/bash
#SBATCH --job-name=exp107-cache-pipeline
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out

# Single coordinator job for the exp107 cache pipeline.
# Runs bootstrap/finalize/verify locally and fans shard workers out in
# parallel on the same node.

set -euo pipefail

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

FEATURES_DIR=${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp106}
CACHE_ROOT=${CACHE_ROOT:-$HOME/scratch/birdclef-2026/cache/exp107}
PARALLEL_WORKERS=${PARALLEL_WORKERS:-4}
FOCAL_CACHE_SIZE=${FOCAL_CACHE_SIZE:-0}
UNSUP_CACHE_SIZE=${UNSUP_CACHE_SIZE:-0}
CACHE_SUBSET_SEED=${CACHE_SUBSET_SEED:-42}
INCLUDE_PERCH_FEATURES=${INCLUDE_PERCH_FEATURES:-${INCLUDE_PERCH_LOGITS:-0}}
CACHE_FORCE_REBUILD=${CACHE_FORCE_REBUILD:-0}
CACHE_FORCE_REBUILD_BASE=${CACHE_FORCE_REBUILD_BASE:-0}
VERIFY_CACHE=${VERIFY_CACHE:-1}
PRINT_MANIFEST=${PRINT_MANIFEST:-0}

if [[ "$PARALLEL_WORKERS" -lt 1 ]]; then
  echo "PARALLEL_WORKERS must be >= 1" >&2
  exit 2
fi
if [[ "$PARALLEL_WORKERS" -ge 16 ]]; then
  echo "PARALLEL_WORKERS must stay under 16; got $PARALLEL_WORKERS" >&2
  exit 2
fi
if [[ "$PARALLEL_WORKERS" -gt "${SLURM_CPUS_PER_TASK:-8}" ]]; then
  echo "PARALLEL_WORKERS=$PARALLEL_WORKERS exceeds cpus-per-task=${SLURM_CPUS_PER_TASK:-8}" >&2
  exit 2
fi

perch_flag=()
if [[ "$INCLUDE_PERCH_FEATURES" != "0" ]]; then
  perch_flag=(--include-perch-features)
fi

force_idx_flag=()
if [[ "$CACHE_FORCE_REBUILD" != "0" ]]; then
  force_idx_flag=(--cache-force-rebuild)
fi

force_base_flag=()
if [[ "$CACHE_FORCE_REBUILD_BASE" != "0" ]]; then
  force_base_flag=(--cache-force-rebuild-base)
fi

print_manifest_flag=()
if [[ "$PRINT_MANIFEST" != "0" ]]; then
  print_manifest_flag=(--print-manifest)
fi

echo "=== exp107 cache pipeline coordinator ==="
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== features_dir=$FEATURES_DIR cache_root=$CACHE_ROOT include_perch=$INCLUDE_PERCH_FEATURES ==="
echo "=== focal_cache_size=$FOCAL_CACHE_SIZE unsup_cache_size=$UNSUP_CACHE_SIZE seed=$CACHE_SUBSET_SEED ==="
echo "=== parallel_workers=$PARALLEL_WORKERS verify_cache=$VERIFY_CACHE ==="

export FEATURES_DIR CACHE_ROOT INCLUDE_PERCH_FEATURES CACHE_FORCE_REBUILD_BASE
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "--- Stage 1/4: bootstrap ---"
birdclef-107 cache bootstrap \
  --features-dir "$FEATURES_DIR" \
  --cache-dir "$CACHE_ROOT" \
  "${force_base_flag[@]}"

N_UNITS=$(birdclef-107 cache plan --features-dir "$FEATURES_DIR" --count-only)
echo "--- Stage 2/4: shard workers (n_units=$N_UNITS, parallel=$PARALLEL_WORKERS) ---"
export POLARS_MAX_THREADS=1
export OMP_NUM_THREADS=1
seq 0 $((N_UNITS - 1)) | xargs -P "$PARALLEL_WORKERS" -I{} bash -lc '
  set -euo pipefail
  args=(cache shard --features-dir "$FEATURES_DIR" --cache-dir "$CACHE_ROOT" --task-id "{}")
  if [[ "$INCLUDE_PERCH_FEATURES" != "0" ]]; then
    args+=(--include-perch-features)
  fi
  if [[ "$CACHE_FORCE_REBUILD_BASE" != "0" ]]; then
    args+=(--cache-force-rebuild-base)
  fi
  birdclef-107 "${args[@]}"
'
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "--- Stage 3/4: finalize ---"
birdclef-107 cache finalize \
  --features-dir "$FEATURES_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --focal-cache-size "$FOCAL_CACHE_SIZE" \
  --unsup-cache-size "$UNSUP_CACHE_SIZE" \
  --seed "$CACHE_SUBSET_SEED" \
  "${perch_flag[@]}" \
  "${force_idx_flag[@]}" \
  "${force_base_flag[@]}" \
  "${print_manifest_flag[@]}"

if [[ "$VERIFY_CACHE" != "0" ]]; then
  echo "--- Stage 4/4: verify ---"
  birdclef-107 cache verify \
    --features-dir "$FEATURES_DIR" \
    --cache-dir "$CACHE_ROOT" \
    --focal-cache-size "$FOCAL_CACHE_SIZE" \
    --unsup-cache-size "$UNSUP_CACHE_SIZE" \
    --seed "$CACHE_SUBSET_SEED" \
    "${perch_flag[@]}" \
    "${print_manifest_flag[@]}"
fi

echo "=== exp107 cache pipeline complete ==="
