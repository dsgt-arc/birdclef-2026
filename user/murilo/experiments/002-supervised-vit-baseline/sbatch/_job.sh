#!/bin/bash
#SBATCH --job-name=exp002-vit-baseline
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 002: Supervised ViT Baseline
# Purpose: Train ViT-Base on BirdSet HSN dataset with PyTorch Lightning
#
# Usage:
#   cd user/murilo/experiments/002-supervised-vit-baseline
#   bash sbatch/run.sh
#

set -eu

# ============================================================================
# Determine paths
# ============================================================================

# SLURM copies script to temp location, so use SLURM_SUBMIT_DIR if available
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Experiment 002 - Supervised ViT Baseline"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${REPO_DIR}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo ""
nvidia-smi
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

# Create job-specific venv in TMPDIR for fast local I/O
JOB_VENV_DIR="$TMPDIR/birdset-benchmark-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11

# Activate the virtual environment
source "$JOB_VENV/bin/activate"

# Install the user package (includes all dependencies from pyproject.toml)
echo "  Installing dependencies..."
uv pip install -e "$USER_DIR"

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""

# ============================================================================
# Create output directories
# ============================================================================

cd "$EXPERIMENT_DIR"
mkdir -p logs
mkdir -p outputs/checkpoints
mkdir -p outputs/wandb

# ============================================================================
# Environment variables
# ============================================================================

# Performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Load .env file for W&B API key (if exists)
if [ -f "$EXPERIMENT_DIR/.env" ]; then
    echo "Loading .env file..."
    set -a
    source "$EXPERIMENT_DIR/.env"
    set +a
fi

# ============================================================================
# Log configuration
# ============================================================================

echo "=============================================="
echo "Configuration:"
cat config.yaml
echo "=============================================="
echo ""

# ============================================================================
# Run training
# ============================================================================

echo "Starting training..."
echo ""

srun python train.py --config config.yaml

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
