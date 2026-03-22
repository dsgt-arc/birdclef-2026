#!/bin/bash
#SBATCH --job-name=infer-soundscapes
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Soundscape ONNX inference job.
# Submitted by sbatch/run.sh — do not run directly.

set -eu

MODEL="${MODEL:-all}"
SPLIT="${SPLIT:-labeled}"
# SCRIPTS_DIR is passed via --export from run.sh
SCRIPTS_DIR="${SCRIPTS_DIR:?SCRIPTS_DIR must be set by run.sh}"

INFER_SCRIPT="${SCRIPTS_DIR}/infer_soundscapes.py"

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Soundscape Inference Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Model: ${MODEL}"
echo "Split: ${SPLIT}"
echo "Scripts dir: ${SCRIPTS_DIR}"
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

export UV_CACHE_DIR="$TMPDIR/uv-cache"

JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing dependencies..."
uv pip install onnxruntime librosa soundfile numpy tqdm scikit-learn pandas

echo "  Installing CPU PyTorch..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "  Virtual environment ready!"
echo ""

# Verify
python -c "import onnxruntime; print(f'onnxruntime: {onnxruntime.__version__}')"
python -c "import librosa; print(f'librosa: {librosa.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
echo ""

# ============================================================================
# Run inference
# ============================================================================

echo "Running inference..."
python "$INFER_SCRIPT" --model "$MODEL" --split "$SPLIT"

echo ""
echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
