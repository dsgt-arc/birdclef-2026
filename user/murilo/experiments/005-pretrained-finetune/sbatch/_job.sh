#!/bin/bash
#SBATCH --job-name=exp005-pretrained-finetune
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 005: Pretrained Audio Model Fine-tuning
# Purpose: Fine-tune EAT/BEATs encoders on BirdCLEF 2025
#
# Usage:
#   cd user/murilo/experiments/005-pretrained-finetune
#   bash sbatch/run.sh config/eat_frozen.yaml
#   bash sbatch/run.sh config/eat_finetune.yaml --fast-dev-run
#

set -eu

# Config file and project name (passed via environment variables from run.sh)
CONFIG="${CONFIG:-config/eat_frozen.yaml}"
PROJECT_NAME="${PROJECT_NAME:-ps-dsgt_clef2026-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"

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
echo "Job: Experiment 005 - Pretrained Fine-tuning"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Config: ${CONFIG}"
echo "Project: ${PROJECT_NAME}"
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
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11

# Activate the virtual environment
source "$JOB_VENV/bin/activate"

# Install the user package (includes all dependencies from pyproject.toml)
echo "  Installing dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers for EAT model loading
echo "  Installing transformers..."
uv pip install "transformers>=4.30.0"

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
mkdir -p outputs/wandb

# ============================================================================
# Environment variables
# ============================================================================

# Performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HuggingFace model cache — use project storage to avoid re-downloading per job
export HF_HOME="/storage/project/${PROJECT_NAME}/shared/birdclef-2026/cache/huggingface"
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

# Load .env file for W&B API key from user directory
if [ -f "$USER_DIR/.env" ]; then
    echo "Loading .env from $USER_DIR/.env..."
    set -a
    source "$USER_DIR/.env"
    set +a
fi

# ============================================================================
# Log configuration
# ============================================================================

echo "=============================================="
echo "Configuration (${CONFIG}):"
cat "${CONFIG}"
echo "=============================================="
echo ""

# ============================================================================
# Run training
# ============================================================================

echo "Starting training..."
echo ""

TRAIN_CMD="python train.py --config ${CONFIG} --project-name ${PROJECT_NAME}"
if [[ "${FAST_DEV_RUN}" == "true" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --fast-dev-run"
    echo "  (fast dev run: 5 batches, no W&B)"
fi

srun ${TRAIN_CMD}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
