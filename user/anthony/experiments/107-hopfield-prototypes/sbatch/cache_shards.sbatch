#!/bin/bash
#SBATCH --job-name=exp107-cache-shards
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%A_%a.out

# Submit with an explicit --array=0-N%K from scripts/submit_cache_build.sh.

set -euo pipefail

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

FEATURES_DIR=${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp106}
CACHE_ROOT=${CACHE_ROOT:-$HOME/scratch/birdclef-2026/cache/exp107}
INCLUDE_PERCH_FEATURES=${INCLUDE_PERCH_FEATURES:-${INCLUDE_PERCH_LOGITS:-0}}
CACHE_FORCE_REBUILD_BASE=${CACHE_FORCE_REBUILD_BASE:-0}

perch_flag=()
if [[ "$INCLUDE_PERCH_FEATURES" != "0" ]]; then
  perch_flag=(--include-perch-features)
fi

force_base_flag=()
if [[ "$CACHE_FORCE_REBUILD_BASE" != "0" ]]; then
  force_base_flag=(--cache-force-rebuild-base)
fi

echo "=== Stage 2/3: exp107 cache shard worker ==="
echo "=== Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) ==="
echo "=== features_dir=$FEATURES_DIR cache_root=$CACHE_ROOT include_perch=$INCLUDE_PERCH_FEATURES ==="

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python -u scripts/build_cache_shard.py \
  --features-dir "$FEATURES_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
  "${perch_flag[@]}" \
  "${force_base_flag[@]}"
