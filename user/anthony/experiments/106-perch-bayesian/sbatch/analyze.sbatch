#!/bin/bash
#SBATCH --job-name=106-analyze
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 360
#SBATCH -q inferno
#SBATCH --output=logs/analyze-%j.out

# Run the Luigi analysis pipeline.
# 64 GB RAM for logit covariance (14795^2 accumulator = ~1.75 GB)
# and sparse logistic (sklearn on 14795-d features).
#
# Usage:
#   sbatch sbatch/analyze.sbatch                    # all phases
#   sbatch sbatch/analyze.sbatch --phase 1          # phase 1 only
#   sbatch sbatch/analyze.sbatch --phase 2          # phase 1 + 2

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

# Install analysis dependencies (idempotent — skips if already installed)
uv pip install luigi scikit-learn scipy pydmd filterpy igraph leidenalg --quiet

python -c "from birdclef_106.cli import main; main()" analyze "$@"
