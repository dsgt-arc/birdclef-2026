#!/bin/bash
#SBATCH --job-name=300-focal-sep
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=12G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/focal-sep-%j.out

# PaCMAP + kNN species-separability of all 6 deployed encoders on FOCAL clips,
# mean-pooled per recording, colored by taxonomic class. CPU-only.
#   sbatch sbatch/plot-focal-separability.sbatch
#   CAP=100 sbatch sbatch/plot-focal-separability.sbatch   # more recordings/species

set -euo pipefail
mkdir -p logs artifacts

VENV_DIR="$(realpath ~/scratch/birdclef-2026)/.venv-300"
source "$VENV_DIR/bin/activate"
uv pip install -q pacmap matplotlib

cd "${SLURM_SUBMIT_DIR:-.}"
python scripts/plot_focal_separability.py --cap "${CAP:-60}" --k "${K:-10}"
