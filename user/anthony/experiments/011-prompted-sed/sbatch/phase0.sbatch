#!/bin/bash
#SBATCH --job-name=011-phase0
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t 60
#SBATCH -q inferno
#SBATCH --output=logs/phase0-%j.out

set -euo pipefail

cd ~/arc/birdclef-2026/user/anthony/experiments/011-prompted-sed
mkdir -p logs

module load cuda/12.1.1

venv=~/arc/birdclef-2026/user/anthony/.venv
export PYTHONUNBUFFERED=1

echo "=== Phase 0: Feasibility Gate ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python: $($venv/bin/python --version)"
echo "Start: $(date)"
echo ""

$venv/bin/python scripts/phase0_feasibility.py \
    --output-dir ~/scratch/birdclef-2026/outputs/exp011/phase0-v2

echo ""
echo "End: $(date)"
