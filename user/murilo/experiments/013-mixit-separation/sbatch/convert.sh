#!/bin/bash
#SBATCH --job-name=exp013-mixit-convert
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#SBATCH --output=logs/slurm_convert_%j.log
#
# Experiment 013, Phase 1: Convert Bird MixIT TF1 checkpoint to ONNX
#
# Usage (via launcher):
#   cd user/murilo/experiments/013-mixit-separation
#   bash sbatch/run.sh convert

set -eu

EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
PROJECT_NAME="ps-clef2026_birdclef-0"

MIXIT_DIR="/storage/project/${PROJECT_NAME}/shared/models/bird-mixit"
DATA_DIR="/storage/project/${PROJECT_NAME}/shared/data/birdclef-2026"

echo "=============================================="
echo "Job: Exp 013 Phase 1 - MixIT TF1 → ONNX"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Setup venv
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
export UV_CACHE_DIR="$TMPDIR/uv-cache"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "Installing dependencies (tensorflow + tf2onnx)..."
uv pip install tensorflow tf2onnx onnx onnxruntime soundfile resampy numpy
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs

# Convert
echo "Converting TF1 → ONNX..."
python scripts/convert_mixit_onnx.py \
    --checkpoint-dir "$MIXIT_DIR" \
    --output "$MIXIT_DIR/bird_mixit_4sources.onnx" \
    --validate \
    --test-audio "${DATA_DIR}/train_soundscapes/$(ls "${DATA_DIR}/train_soundscapes/" | head -1)"

echo ""
echo "=============================================="
echo "Conversion complete! $(date)"
echo "ONNX model: $MIXIT_DIR/bird_mixit_4sources.onnx"
echo "=============================================="
