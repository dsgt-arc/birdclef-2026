#!/bin/bash
#SBATCH --job-name=300-paper-sep
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=10G
#SBATCH -p cpu-small
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/paper-sep-%j.out

# Paper-ready encoder-separability figure: slim 1x6 PaCMAP filmstrip, 8-species subset,
# colored by species, fine dots + faint 4x4 grid. Vector PDF + 600-DPI PNG. CPU-only, ~minutes.
#   sbatch sbatch/plot-paper-separability.sbatch
#   CAP=40 MARKER=2 HEIGHT=2.0 LAYOUT=grid sbatch sbatch/plot-paper-separability.sbatch

set -euo pipefail
mkdir -p logs artifacts

VENV_DIR="$(realpath ~/scratch/birdclef-2026)/.venv-300"
source "$VENV_DIR/bin/activate"
uv pip install -q pacmap matplotlib

cd "${SLURM_SUBMIT_DIR:-.}"
python scripts/plot_paper_separability.py \
    --cap "${CAP:-60}" --marker-size "${MARKER:-3}" --height "${HEIGHT:-1.9}" \
    --layout "${LAYOUT:-strip}"
