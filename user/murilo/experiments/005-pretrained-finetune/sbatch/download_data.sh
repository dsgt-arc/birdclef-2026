#!/bin/bash
#SBATCH --job-name=exp005-download
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Download BirdCLEF competition data (CPU-only, no GPU needed).
#
# Usage:
#   cd user/murilo/experiments/005-pretrained-finetune
#   sbatch sbatch/download_data.sh                              # default: birdclef-2025
#   sbatch sbatch/download_data.sh --export=COMPETITION=birdclef-2026  # for 2026
#

set -eu

COMPETITION="${COMPETITION:-birdclef-2025}"

# ============================================================================
# Determine paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Download ${COMPETITION} Dataset"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Experiment: ${EXPERIMENT_DIR}"
echo "Competition: ${COMPETITION}"
echo ""

# ============================================================================
# Setup minimal venv (just kaggle + dotenv, no torch needed)
# ============================================================================

echo "Setting up minimal venv for download..."

JOB_VENV_DIR="$TMPDIR/birdclef-download-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

uv pip install kaggle python-dotenv
echo "  Venv ready: $JOB_VENV"
echo ""

# ============================================================================
# Run download
# ============================================================================

cd "$EXPERIMENT_DIR"
python scripts/download_birdclef_data.py --competition "$COMPETITION"

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Download completed!"
echo "=============================================="
