#!/bin/bash
#SBATCH --job-name=exp038-pretrain
#SBATCH --account=paceship-clef2026_birdclef
# --partition / --output / --export / --job-name are set by run.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 038 Stage 1 — broad Neotropical XC supervised pretraining (EffNetV2-S IN21k, SED-mirror).
# GPU worker for run.sh (do not invoke directly; use `bash sbatch/run.sh <config>` which sbatches
# this). Builds a per-job $TMPDIR venv (CLAUDE.md: never install into home's 20 GB quota), then
# runs train.py over the curated 160K-recording / 2,683-species corpus.
#
# Usage (via the launcher):
#   cd user/murilo/experiments/038-xc-pretraining
#   bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml                 # primary
#   bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml --fast-dev-run  # gradient-flow check
set -eu

CONFIG="${CONFIG:-configs/pretrain_cellA_dualbce.yaml}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
GPU_CHECK="${GPU_CHECK:-0}"
NUM_WORKERS="${NUM_WORKERS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
ACCUMULATE="${ACCUMULATE:-}"

# $SLURM_SUBMIT_DIR = the experiment dir (where run.sh called sbatch). NOT $BASH_SOURCE — sbatch
# spools a copy of this script ([[feedback_slurm_script_conventions]]).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Exp 038 Stage 1 — XC supervised pretraining"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | $(date)"
echo "Config: ${CONFIG}"
echo "Git: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown))"
echo ""
nvidia-smi
echo ""

# --- per-job venv on $TMPDIR (fast local NVMe; never home) ---
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
mkdir -p "$TMPDIR/birdclef-$SLURM_JOB_ID"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing experiment deps..."
uv pip install -e "$EXPERIMENT_DIR"
echo "  Installing CUDA torch/torchvision/torchaudio (cu121)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
echo "  Venv ready."
echo ""
python -c "import torch, timm; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} | timm {timm.__version__}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs outputs/wandb
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

if [ -f "$USER_DIR/.env" ]; then
    set -a
    source "$USER_DIR/.env"
    set +a
fi

echo "=============================================="
echo "Config (${CONFIG}):"
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
    echo "  (gpu-check: ${GPU_CHECK} epoch(s), GPU monitor, no W&B/checkpoints)"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --num-workers ${NUM_WORKERS}"
fi
if [[ -n "${BATCH_SIZE}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --batch-size ${BATCH_SIZE}"
fi
if [[ -n "${ACCUMULATE}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --accumulate-grad-batches ${ACCUMULATE}"
fi

echo "Starting pretraining..."
srun ${TRAIN_CMD}

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="
