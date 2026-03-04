#!/bin/bash
#SBATCH --job-name=infer-protoclr
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
# Inference: Extract embeddings and predictions from fine-tuned models
#
# Usage:
#   cd user/murilo/inference
#   bash sbatch/run.sh config/protoclr_lora_attn.yaml --source train_audio
#   bash sbatch/run.sh config/protoclr_lora_attn.yaml --source all
#

set -eu

# Environment variables (passed from run.sh)
CONFIG="${CONFIG:-config/protoclr_lora_attn.yaml}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"
SOURCE="${SOURCE:-all}"
MAX_FILES="${MAX_FILES:-}"
NUM_WORKERS="${NUM_WORKERS:-}"

# ============================================================================
# Determine paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$EXPERIMENT_DIR")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Inference - Embedding & Prediction Extraction"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Config: ${CONFIG}"
echo "Source: ${SOURCE}"
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

# Use a per-job uv cache on the same filesystem as the venv
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
if grep -q "^lora:" "${CONFIG}"; then
    python -c "import peft; print(f'PEFT: {peft.__version__}')"
fi

# Verify einops (needed by CvT)
python -c "import einops; print(f'einops: {einops.__version__}')"
echo ""

# ============================================================================
# Create output directories
# ============================================================================

cd "$EXPERIMENT_DIR"
mkdir -p outputs

# ============================================================================
# Environment variables
# ============================================================================

# Performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HuggingFace cache -- use project storage for model download
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"
echo ""

# ============================================================================
# Log configuration
# ============================================================================

echo "=============================================="
echo "Configuration (${CONFIG}):"
cat "${CONFIG}"
echo "=============================================="
echo ""

# ============================================================================
# Run inference
# ============================================================================

echo "Starting inference..."
echo ""

INFER_CMD="python infer.py --config ${CONFIG} --source ${SOURCE}"
if [[ -n "${MAX_FILES}" ]]; then
    INFER_CMD="${INFER_CMD} --max-files ${MAX_FILES}"
    echo "  (max files: ${MAX_FILES})"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
    INFER_CMD="${INFER_CMD} --num-workers ${NUM_WORKERS}"
    echo "  (num_workers override: ${NUM_WORKERS})"
fi

srun ${INFER_CMD}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
