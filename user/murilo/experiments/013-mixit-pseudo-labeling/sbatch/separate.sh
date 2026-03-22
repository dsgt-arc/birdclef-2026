#!/bin/bash
#SBATCH --job-name=exp013-mixit-separate
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#SBATCH --output=logs/slurm_separate_%A_%a.log
#
# Experiment 013, Phase 2: Run Bird MixIT separation on soundscapes
# Default: GPU mode with SLURM job array support for distributed processing
#
# Usage (via launcher):
#   cd user/murilo/experiments/013-mixit-pseudo-labeling
#   bash sbatch/run.sh separate                          # 1 GPU (default)
#   bash sbatch/run.sh separate --num-workers 8          # 8 GPUs in parallel
#   bash sbatch/run.sh separate --labeled                # labeled only, 1 GPU
#   bash sbatch/run.sh separate --labeled --num-workers 4 # labeled, 4 GPUs
#   bash sbatch/run.sh separate --cpu                    # CPU fallback

set -eu

MODE="${MODE:-unlabeled}"
USE_GPU="${USE_GPU:-true}"
NUM_WORKERS="${NUM_WORKERS:-1}"

# SLURM array task info (set by SLURM when using --array)
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"

EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
PROJECT_NAME="ps-clef2026_birdclef-0"

MIXIT_MODEL="/storage/project/${PROJECT_NAME}/shared/models/bird-mixit/bird_mixit_4sources.onnx"
DATA_DIR="/storage/project/${PROJECT_NAME}/shared/data/birdclef-2026"
SOUNDSCAPE_DIR="${DATA_DIR}/train_soundscapes"

echo "=============================================="
echo "Job: Exp 013 Phase 2 - MixIT Separation"
echo "=============================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
echo "Worker: ${WORKER_ID}/${NUM_WORKERS}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Mode: ${MODE}"
echo "GPU: ${USE_GPU}"
echo ""

if [[ "${USE_GPU}" == "true" ]]; then
    nvidia-smi
    echo ""
fi

# Setup venv — each array task gets its own TMPDIR and venv (fully isolated)
JOB_VENV_DIR="$TMPDIR/birdclef-${SLURM_JOB_ID:-$$}"
JOB_VENV="$JOB_VENV_DIR/.venv"
export UV_CACHE_DIR="$TMPDIR/uv-cache"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "Installing dependencies..."
if [[ "${USE_GPU}" == "true" ]]; then
    # onnxruntime-gpu with bundled CUDA/cuDNN libs
    # See: https://onnxruntime.ai/docs/install/
    uv pip install "onnxruntime-gpu[cuda,cudnn]" soundfile resampy numpy tqdm
else
    uv pip install onnxruntime soundfile resampy numpy tqdm
fi
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs

# Output to shared project storage (persistent, team-accessible)
MIXIT_OUTPUT_BASE="/storage/project/${PROJECT_NAME}/shared/mixit/pretrained"

# Verify model exists
if [ ! -f "$MIXIT_MODEL" ]; then
    echo "ERROR: ONNX model not found at $MIXIT_MODEL"
    echo "Run Phase 1 first: bash sbatch/run.sh convert"
    exit 1
fi

# Build flags for python script
EXTRA_FLAGS=""
if [[ "${USE_GPU}" == "true" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --gpu"
fi
if [[ "${USE_GPU}" != "true" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --cpu"
fi
if [[ "${NUM_WORKERS}" -gt 1 ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --worker-id ${WORKER_ID} --num-workers ${NUM_WORKERS}"
fi

# Run separation based on mode
case "$MODE" in
    unlabeled)
        echo "Separating unlabeled soundscapes..."
        echo "Output: ${MIXIT_OUTPUT_BASE}/soundscapes/"
        python scripts/separate_soundscapes.py \
            --model "$MIXIT_MODEL" \
            --soundscape-dir "$SOUNDSCAPE_DIR" \
            --data-dir "$DATA_DIR" \
            --output-dir "${MIXIT_OUTPUT_BASE}/soundscapes" \
            --exclude-labeled \
            --num-threads "$SLURM_CPUS_PER_TASK" \
            ${EXTRA_FLAGS}
        ;;
    labeled)
        echo "Separating labeled soundscapes (for validation)..."
        echo "Output: ${MIXIT_OUTPUT_BASE}/soundscapes_labeled/"
        python scripts/separate_soundscapes.py \
            --model "$MIXIT_MODEL" \
            --soundscape-dir "$SOUNDSCAPE_DIR" \
            --data-dir "$DATA_DIR" \
            --output-dir "${MIXIT_OUTPUT_BASE}/soundscapes_labeled" \
            --only-labeled \
            --num-threads "$SLURM_CPUS_PER_TASK" \
            ${EXTRA_FLAGS}
        ;;
    test)
        echo "Test run (10 files)..."
        echo "Output: ${MIXIT_OUTPUT_BASE}/soundscapes/"
        python scripts/separate_soundscapes.py \
            --model "$MIXIT_MODEL" \
            --soundscape-dir "$SOUNDSCAPE_DIR" \
            --data-dir "$DATA_DIR" \
            --output-dir "${MIXIT_OUTPUT_BASE}/soundscapes" \
            --max-files 10 \
            --num-threads "$SLURM_CPUS_PER_TASK" \
            ${EXTRA_FLAGS}
        ;;
    *)
        echo "Unknown mode: $MODE (expected: unlabeled, labeled, test)"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Worker ${WORKER_ID}/${NUM_WORKERS} complete! $(date)"
echo "=============================================="
