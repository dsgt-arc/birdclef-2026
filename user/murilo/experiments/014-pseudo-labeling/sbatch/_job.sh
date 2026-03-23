#!/bin/bash
#SBATCH --job-name=exp014-pseudo-label
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 014: Teacher pseudo-labeling with GPU ONNX inference.
# Submitted by sbatch/run.sh — do not run directly.

set -eu

CONFIG="${CONFIG:-config/mode_a.yaml}"
SPLIT="${SPLIT:-labeled}"
MAX_FILES="${MAX_FILES:-}"

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
echo "Job: Experiment 014 — Pseudo-Labeling"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Config: ${CONFIG}"
echo "Split: ${SPLIT}"
echo "Max files: ${MAX_FILES:-all}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo ""
nvidia-smi
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
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing PyTorch with CUDA..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

# Verify
python -c "import onnxruntime as ort; print(f'onnxruntime: {ort.__version__}, providers: {ort.get_available_providers()}')"
python -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

# ============================================================================
# Run pseudo-labeling
# ============================================================================

cd "$EXPERIMENT_DIR"
mkdir -p logs

EXTRA_FLAGS=""
if [[ -n "${MAX_FILES}" ]]; then
    EXTRA_FLAGS="--max-files ${MAX_FILES}"
fi

echo "Running pseudo-labeling..."
python pseudo_label.py --config "${CONFIG}" --split "${SPLIT}" ${EXTRA_FLAGS}

echo ""
echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
