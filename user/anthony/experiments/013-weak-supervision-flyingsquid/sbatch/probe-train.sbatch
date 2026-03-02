#!/bin/bash
#SBATCH --job-name=013-probe-train
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t240 -qinferno
#SBATCH --output=logs/probe-train-%j.out

# Usage:
#   sbatch sbatch/probe-train.sbatch birdnet
#   sbatch sbatch/probe-train.sbatch perch

set -eu

source="${1:?Usage: sbatch probe-train.sbatch <birdnet|perch>}"

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"

echo "=== Probe training: $source ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

"$venv/bin/weaksup-probe" all "$source" \
    --gamma 2.0 \
    --lr 1e-3 \
    --epochs 50 \
    --batch-size 16384 \
    --data-dir "$scratch_dir/data/birdclef-2025" \
    --output-dir "$scratch_dir/outputs/exp013/probes"

echo ""
echo "=== Done: $(date) ==="
