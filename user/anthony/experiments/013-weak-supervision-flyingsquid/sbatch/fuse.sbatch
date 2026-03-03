#!/bin/bash
#SBATCH --job-name=013-fuse
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t240 -qinferno
#SBATCH --array=0-1
#SBATCH --output=logs/fuse-%A_%a.out

# Fuse labels for focal (task 0) and soundscape (task 1) in parallel.
# CPU-only. Each task runs independently.
# Usage:
#   sbatch sbatch/fuse.sbatch
#
# Submission workflow with dependencies:
#   JOB_PROBE=$(sbatch sbatch/probe-indices.sbatch | awk '{print $4}')
#   JOB_FUSE=$(sbatch --dependency=afterok:$JOB_PROBE sbatch/fuse.sbatch | awk '{print $4}')
#   sbatch --dependency=afterok:$JOB_FUSE sbatch/evaluate.sbatch

set -euo pipefail

SPLITS=(focal soundscape)
SPLIT="${SPLITS[$SLURM_ARRAY_TASK_ID]}"

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Fusing $SPLIT ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir" --quiet

"$venv/bin/weaksup-fuse" \
    --split "$SPLIT" \
    --probe-dir "$scratch_dir/outputs/exp013/probes" \
    --data-dir "$scratch_dir/data/birdclef-2025" \
    --exp010-dir "$scratch_dir/outputs/exp010/labels" \
    --output-dir "$scratch_dir/outputs/exp013/fused" \
    --n-workers 8

echo ""
echo "=== Done $SPLIT: $(date) ==="
