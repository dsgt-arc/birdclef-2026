#!/bin/bash
#SBATCH --job-name=exp033-cache-inference
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_cache_inference_%j.log
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 033 Phase 0.1: cache distilled-sed-kd-5fold ONNX predictions on all
# 10,658 train_soundscapes files. Output:
#   shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/fold{0..4}/logits.parquet
#
# Walltime budget: 2hr (actual expected ~30-60 min on rtx6000).
#
# Usage:
#   cd user/murilo/experiments/033-own-teacher-pl
#   sbatch sbatch/cache_inference.sh
#
#   # Smoke test (5 files, override via env):
#   LIMIT=5 SUFFIX=_smoke sbatch sbatch/cache_inference.sh

set -eu

LIMIT="${LIMIT:-}"
SUFFIX="${SUFFIX:-}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Exp 033 Phase 0.1: cache_inference (T2 teacher)"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Git commit:    $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Limit:         ${LIMIT:-<all>}"
echo "Output suffix: ${SUFFIX:-<none>}"
echo ""
nvidia-smi
echo ""

# Setup venv
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

uv pip install -e "$EXPERIMENT_DIR"
uv pip install "torch>=2.0.0,<2.11" "torchaudio>=2.0.0,<2.11" \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

EXTRA_ARGS=""
if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS="--limit ${LIMIT}"
fi
if [ -n "${SUFFIX}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --output-suffix ${SUFFIX}"
fi

echo "=============================================="
echo "Running cache_inference.py..."
echo "=============================================="
srun python scripts/cache_inference.py ${EXTRA_ARGS}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
