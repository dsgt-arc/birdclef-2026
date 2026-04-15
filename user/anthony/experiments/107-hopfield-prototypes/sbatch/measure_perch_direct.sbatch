#!/bin/bash
#SBATCH --job-name=exp107-perch-direct
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o logs/%x-%j.out

# Diagnostic: AUROC of Perch's direct logits (projected via exp 106's
# mapping) on the labeled SC windows. No training. CPU-only.

set -euo pipefail

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

echo "=== Measuring Perch direct AUROC ==="
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="

export PYTHONUNBUFFERED=1

python -u scripts/measure_perch_direct.py
