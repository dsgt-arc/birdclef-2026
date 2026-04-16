#!/bin/bash
#SBATCH --job-name=exp107-cache-finalize
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

FEATURES_DIR=${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp106}
CACHE_ROOT=${CACHE_ROOT:-$HOME/scratch/birdclef-2026/cache/exp107}
FOCAL_CACHE_SIZE=${FOCAL_CACHE_SIZE:-0}
UNSUP_CACHE_SIZE=${UNSUP_CACHE_SIZE:-0}
CACHE_SUBSET_SEED=${CACHE_SUBSET_SEED:-42}
INCLUDE_PERCH_FEATURES=${INCLUDE_PERCH_FEATURES:-${INCLUDE_PERCH_LOGITS:-0}}
CACHE_FORCE_REBUILD=${CACHE_FORCE_REBUILD:-0}
CACHE_FORCE_REBUILD_BASE=${CACHE_FORCE_REBUILD_BASE:-0}
PRINT_MANIFEST=${PRINT_MANIFEST:-0}

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

echo "=== Stage 3/3: finalize exp107 cache ==="
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== features_dir=$FEATURES_DIR cache_root=$CACHE_ROOT include_perch=$INCLUDE_PERCH_FEATURES ==="
echo "=== focal_cache_size=$FOCAL_CACHE_SIZE unsup_cache_size=$UNSUP_CACHE_SIZE seed=$CACHE_SUBSET_SEED ==="

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u scripts/build_cache_finalize.py \
  --features-dir "$FEATURES_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --focal-cache-size "$FOCAL_CACHE_SIZE" \
  --unsup-cache-size "$UNSUP_CACHE_SIZE" \
  --seed "$CACHE_SUBSET_SEED" \
  "${perch_flag[@]}" \
  "${force_idx_flag[@]}" \
  "${force_base_flag[@]}" \
  "${print_manifest_flag[@]}"
