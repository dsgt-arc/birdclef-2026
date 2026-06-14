#!/bin/bash
#SBATCH --job-name=300-sweep-lam
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/sweep-lambda-%j.out

# Select ridge-head λ by held-out focal->soundscape AUROC (focal-only fit, eval on the
# 739 soundscapes). Streams focal once -> artifacts/head_lambda_sweep.json.

set -euo pipefail
mkdir -p logs artifacts

export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR"
source sbatch/_setup-venv.sh

python -m birdclef_300.deploy.sweep_lambda \
    --features-dir "${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp106}" \
    --mapping "${MAPPING:-artifacts/perch_species_mapping.json}" \
    --out "${OUT:-artifacts/head_lambda_sweep.json}"
