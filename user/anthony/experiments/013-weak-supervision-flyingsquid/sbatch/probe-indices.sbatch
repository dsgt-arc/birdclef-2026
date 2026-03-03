#!/bin/bash
#SBATCH --job-name=013-probe-indices
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/probe-indices-%j.out

# Train species probe on acoustic indices (4 features).
# CPU-only, completes in minutes.
# Usage:
#   sbatch sbatch/probe-indices.sbatch

set -euo pipefail

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Indices probe training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir" --quiet

"$venv/bin/weaksup-probe" species indices \
    --gamma 2.0 \
    --epochs 50 \
    --batch-size 16384 \
    --data-dir "$scratch_dir/data/birdclef-2025" \
    --output-dir "$scratch_dir/outputs/exp013/probes"

echo ""
echo "=== Done: $(date) ==="
