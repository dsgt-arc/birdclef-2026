#!/bin/bash
#SBATCH --job-name=precompute-losses
#SBATCH --account=paceship-clef2026_birdclef
# --partition is set by run.sh (default: gpu-rtx6000,gpu-v100)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Pre-compute per-sample losses for the model attention explorer.
#
# Usage:
#   cd user/murilo/analysis
#   bash sbatch/run.sh                              # v5j defaults
#   bash sbatch/run.sh --config path/to/config.yaml --checkpoint path/to/ckpt

set -eu

CONFIG="${CONFIG:?CONFIG must be set}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT must be set}"
OUTPUT="${OUTPUT:-losses.parquet}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"
DATA_SPLIT="${DATA_SPLIT:-val}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ANALYSIS_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$ANALYSIS_DIR")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Job: Pre-compute Losses for Model Explorer"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo "Git commit: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
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
uv pip install -e "$USER_DIR"

echo "  Installing PyTorch with CUDA..."
uv pip install "torch>=2.0.0,<2.11" "torchvision>=0.15,<0.26" "torchaudio>=2.0.0,<2.11" --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""

cd "$ANALYSIS_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

if [ -f "$USER_DIR/.env" ]; then
    set -a
    source "$USER_DIR/.env"
    set +a
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "=============================================="
echo "Starting pre-computation..."
echo "=============================================="
echo ""

srun python precompute_losses.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --data-split "${DATA_SPLIT}"

# Triage enrichment is val-only (joins with train.csv to derive co-occurrence
# categories that don't make sense for the train split). Skip on train mode.
if [ "${DATA_SPLIT}" = "val" ]; then
    echo ""
    echo "=============================================="
    echo "Running triage enrichment..."
    echo "=============================================="
    set +e
    python triage.py --losses "${OUTPUT}"
    TRIAGE_RC=$?
    set -e
    if [ $TRIAGE_RC -ne 0 ]; then
        echo "WARNING: triage.py exited with code $TRIAGE_RC — losses parquet is fine, re-run triage manually"
    fi
else
    echo ""
    echo "Train split: skipping triage (val-only)."
fi

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo ""
echo "Output files:"
TRIAGE_OUTPUT="$(dirname "${OUTPUT}")/$(basename "${OUTPUT}" | sed 's/^losses_/triage_/')"
ls -lh \
    "${OUTPUT}" \
    "${OUTPUT%.parquet}_logits.npy" \
    "${OUTPUT%.parquet}_perframe_logits.npy" \
    "${OUTPUT%.parquet}_perframe_att.npy" \
    "${TRIAGE_OUTPUT}" 2>/dev/null
echo "=============================================="
