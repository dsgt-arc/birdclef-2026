#!/bin/bash
#SBATCH --job-name=013-evaluate
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/evaluate-%j.out

# Evaluate fused labels for both focal and soundscape splits.
# CPU-only, fast.
# Usage:
#   sbatch sbatch/evaluate.sbatch

set -euo pipefail

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir" --quiet

fused_dir="$scratch_dir/outputs/exp013/fused"
data_dir="$scratch_dir/data/birdclef-2025"
eval_dir="$scratch_dir/outputs/exp013/eval"

echo "=== Evaluating focal ==="
"$venv/bin/weaksup-evaluate" \
    --split focal \
    --fused-dir "$fused_dir" \
    --data-dir "$data_dir" \
    --output-dir "$eval_dir"

echo ""

echo "=== Evaluating soundscape ==="
"$venv/bin/weaksup-evaluate" \
    --split soundscape \
    --fused-dir "$fused_dir" \
    --data-dir "$data_dir" \
    --output-dir "$eval_dir"

echo ""
echo "=== Done: $(date) ==="
