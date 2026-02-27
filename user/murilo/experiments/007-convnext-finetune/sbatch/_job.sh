#!/bin/bash
#SBATCH --job-name=exp007-convnext-finetune
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 007: ConvNeXt-Base LoRA Fine-Tuning on BirdCLEF 2025
# Purpose: Apply PEFT LoRA to ConvNeXt-Base MLP layers (~1.8% trainable params)
#
# Usage:
#   cd user/murilo/experiments/007-convnext-finetune
#   bash sbatch/run.sh config/convnext_lora_mlp.yaml
#   bash sbatch/run.sh config/convnext_lora_mlp_mlphead.yaml --fast-dev-run
#

set -eu

# Config file and project name (passed via environment variables from run.sh)
CONFIG="${CONFIG:-config/convnext_lora_mlp.yaml}"
PROJECT_NAME="${PROJECT_NAME:-ps-dsgt_clef2026-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
GPU_CHECK="${GPU_CHECK:-0}"
NUM_WORKERS="${NUM_WORKERS:-}"

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
echo "Job: Experiment 007 - ConvNeXt LoRA Fine-Tuning"
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

# Use a per-job uv cache on the same filesystem as the venv (TMPDIR = local NVMe).
# The global uv cache lives on scratch (XDG_CACHE_HOME) which is a different
# filesystem; uv's hardlink/copy from scratch → /tmp silently produces empty
# directory stubs (e.g., tokenizers .so missing). A local cache avoids this and
# is faster. The trade-off is re-downloading packages per job, but install time
# is dominated by compilation, not downloads.
export UV_CACHE_DIR="$TMPDIR/uv-cache"

# Create job-specific venv in TMPDIR for fast local I/O
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11

# Activate the virtual environment
source "$JOB_VENV/bin/activate"

# Install dependencies
echo "  Installing dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Verify PEFT
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Verify timm
python -c "import timm; print(f'timm: {timm.__version__}')"
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

# timm model cache — use project storage to avoid re-downloading per job
export TORCH_HOME="/storage/project/${PROJECT_NAME}/shared/birdclef-2026/cache/torch"
mkdir -p "$TORCH_HOME"
echo "TORCH_HOME: $TORCH_HOME"

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
if [[ "${GPU_CHECK}" -gt 0 ]]; then
    TRAIN_CMD="${TRAIN_CMD} --gpu-check ${GPU_CHECK}"
    echo "  (GPU check mode: ${GPU_CHECK} epoch(s), no W&B)"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --num-workers ${NUM_WORKERS}"
    echo "  (num_workers override: ${NUM_WORKERS})"
fi

srun ${TRAIN_CMD}

# ============================================================================
# Save structured metrics to artifacts/ (skip for fast-dev-run and gpu-check)
# ============================================================================

if [[ "${FAST_DEV_RUN}" != "true" && "${GPU_CHECK}" -eq 0 ]]; then
    echo ""
    echo "Saving structured metrics to artifacts/..."
    python save_metrics.py --config "${CONFIG}"
fi

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
