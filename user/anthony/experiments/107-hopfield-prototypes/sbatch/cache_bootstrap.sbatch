#!/bin/bash
#SBATCH --job-name=exp107-cache-bootstrap
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
CACHE_FORCE_REBUILD_BASE=${CACHE_FORCE_REBUILD_BASE:-0}
PRINT_METADATA=${PRINT_METADATA:-0}

force_base_flag=()
if [[ "$CACHE_FORCE_REBUILD_BASE" != "0" ]]; then
  force_base_flag=(--cache-force-rebuild-base)
fi

print_metadata_flag=()
if [[ "$PRINT_METADATA" != "0" ]]; then
  print_metadata_flag=(--print-metadata)
fi

echo "=== Stage 1/3: bootstrap exp107 cache metadata ==="
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== features_dir=$FEATURES_DIR cache_root=$CACHE_ROOT ==="

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u scripts/build_cache_bootstrap.py \
  --features-dir "$FEATURES_DIR" \
  --cache-dir "$CACHE_ROOT" \
  "${force_base_flag[@]}" \
  "${print_metadata_flag[@]}"
