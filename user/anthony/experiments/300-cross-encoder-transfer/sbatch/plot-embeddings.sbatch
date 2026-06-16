#!/bin/bash
#SBATCH --job-name=300-pacmap
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/pacmap-%j.out

# PaCMAP 2D projection of all 6 deployed encoders on the labeled soundscapes,
# colored by taxonomic class. Reads the frozen-embedding cache; CPU-only, ~minutes.
#   sbatch sbatch/plot-embeddings.sbatch

set -euo pipefail
mkdir -p logs artifacts

VENV_DIR="$(realpath ~/scratch/birdclef-2026)/.venv-300"
source "$VENV_DIR/bin/activate"

# Extra viz deps (not in the shared extraction/probe venv); idempotent on scratch.
uv pip install -q pacmap matplotlib

cd "${SLURM_SUBMIT_DIR:-.}"
python scripts/plot_embeddings_pacmap.py
