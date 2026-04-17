#!/bin/bash
#SBATCH --job-name=exp107-verify-cache
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
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
PRINT_MANIFEST=${PRINT_MANIFEST:-0}
SAMPLE_ROWS=${SAMPLE_ROWS:-8}

perch_flag=()
if [[ "$INCLUDE_PERCH_FEATURES" != "0" ]]; then
  perch_flag=(--include-perch-features)
fi

print_manifest_flag=()
if [[ "$PRINT_MANIFEST" != "0" ]]; then
  print_manifest_flag=(--print-manifest)
fi

echo "=== Verify exp107 cache ==="
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== features_dir=$FEATURES_DIR cache_root=$CACHE_ROOT include_perch=$INCLUDE_PERCH_FEATURES ==="
echo "=== focal_cache_size=$FOCAL_CACHE_SIZE unsup_cache_size=$UNSUP_CACHE_SIZE seed=$CACHE_SUBSET_SEED sample_rows=$SAMPLE_ROWS ==="

export PYTHONUNBUFFERED=1

python -u scripts/verify_cache.py \
  --features-dir "$FEATURES_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --focal-cache-size "$FOCAL_CACHE_SIZE" \
  --unsup-cache-size "$UNSUP_CACHE_SIZE" \
  --seed "$CACHE_SUBSET_SEED" \
  --sample-rows "$SAMPLE_ROWS" \
  "${perch_flag[@]}" \
  "${print_manifest_flag[@]}"
