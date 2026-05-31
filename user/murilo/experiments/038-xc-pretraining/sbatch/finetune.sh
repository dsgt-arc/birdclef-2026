#!/bin/bash
#SBATCH --job-name=exp038-finetune
#SBATCH --account=paceship-clef2026_birdclef
# --partition / --output / --export / --job-name are set by run_finetune.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 038 Stage 2 — fine-tune an XC-pretrained EffNetV2-S backbone on the 234 targets (KD-SED recipe).
# GPU worker for run_finetune.sh (do not invoke directly; use `bash sbatch/run_finetune.sh <config>
# --fold N`). Builds a per-job $TMPDIR venv (CLAUDE.md: never install into home's 20 GB quota), then
# runs train_finetune.py (the 0.906 recipe with the EffNetV2-S backbone warm-started from Stage 1).
set -eu

CONFIG="${CONFIG:-configs/finetune_cellA.yaml}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
FOLD="${FOLD:-}"
NUM_WORKERS="${NUM_WORKERS:-}"

# $SLURM_SUBMIT_DIR = the experiment dir (where run_finetune.sh called sbatch). NOT $BASH_SOURCE —
# sbatch spools a copy of this script ([[feedback_slurm_script_conventions]]).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Exp 038 Stage 2 — XC-pretrained backbone fine-tune (KD-SED)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | $(date)"
echo "Config: ${CONFIG} | Fold: ${FOLD:-<none>}"
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

echo "  Installing experiment deps (incl. onnxruntime-gpu for the Perch KD teacher)..."
uv pip install -e "$EXPERIMENT_DIR"
echo "  Installing CUDA torch/torchvision/torchaudio (cu121)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
echo "  Venv ready."
echo ""
python -c "import torch, timm, onnxruntime; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} | timm {timm.__version__} | ort {onnxruntime.__version__}')"
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

TRAIN_CMD="python train_finetune.py --config ${CONFIG} --project-name ${PROJECT_NAME}"
if [[ -n "${FOLD}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --fold ${FOLD}"
fi
if [[ "${FAST_DEV_RUN}" == "true" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --fast-dev-run"
    echo "  (fast dev run: 5 batches, no W&B)"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --num-workers ${NUM_WORKERS}"
fi

echo "Starting fine-tune..."
srun ${TRAIN_CMD}

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="
