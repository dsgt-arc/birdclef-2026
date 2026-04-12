#!/bin/bash
#SBATCH --job-name=exp027-mtf
#SBATCH --account=paceship-clef2026_birdclef
# --partition is set by run.sh (default: gpu-rtx6000,gpu-v100; override with --gpu)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 027: Model Training Fixes (additive MixUp + CE loss)
#
# Usage:
#   cd user/murilo/experiments/027-model-training-fixes
#   bash sbatch/run.sh configs/v10a_addmix_ce.yaml
#   bash sbatch/run.sh configs/v10a_addmix_ce.yaml --fast-dev-run

set -eu

CONFIG="${CONFIG:-configs/v10a_addmix_ce.yaml}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
GPU_CHECK="${GPU_CHECK:-0}"
NUM_WORKERS="${NUM_WORKERS:-}"
FOLD="${FOLD:-}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Job: Experiment 027 - Model Training Fixes"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Config: ${CONFIG}"
echo "Git commit: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Git branch: $(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo 'unknown')"
echo ""
nvidia-smi
echo ""

# Setup venv
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing PyTorch with CUDA..."
uv pip install "torch>=2.0.0,<2.11" "torchvision>=0.15,<0.26" "torchaudio>=2.0.0,<2.11" --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs outputs/wandb

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

if [ -f "$USER_DIR/.env" ]; then
    set -a
    source "$USER_DIR/.env"
    set +a
fi

echo "=============================================="
echo "Configuration (${CONFIG}):"
cat "${CONFIG}"
echo "=============================================="
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
fi
if [[ -n "${FOLD}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --fold ${FOLD}"
    echo "  (fold ${FOLD} of 4-fold CV)"
fi

echo "Starting training..."
srun ${TRAIN_CMD}

# Save structured metrics (skip for fast-dev-run and gpu-check)
if [[ "${FAST_DEV_RUN}" != "true" && "${GPU_CHECK}" -eq 0 ]]; then
    echo ""
    echo "Saving structured metrics to artifacts/..."
    SAVE_CMD="python save_metrics.py --config ${CONFIG}"
    if [[ -n "${FOLD}" ]]; then
        SAVE_CMD="${SAVE_CMD} --fold ${FOLD}"
    fi
    ${SAVE_CMD}
fi

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
