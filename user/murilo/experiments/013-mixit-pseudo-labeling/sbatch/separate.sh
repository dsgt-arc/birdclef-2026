#!/bin/bash
#SBATCH --job-name=exp013-mixit-separate
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#SBATCH --output=logs/slurm_separate_%j.log
#
# Experiment 013, Phase 2: Run Bird MixIT separation on soundscapes
#
# Usage (via launcher):
#   cd user/murilo/experiments/013-mixit-pseudo-labeling
#   bash sbatch/run.sh separate              # all unlabeled
#   bash sbatch/run.sh separate --labeled    # labeled only (validation)
#   bash sbatch/run.sh separate --test       # 10 files (smoke test)

set -eu

MODE="${MODE:-unlabeled}"
USE_GPU="${USE_GPU:-false}"

EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
PROJECT_NAME="ps-clef2026_birdclef-0"

MIXIT_MODEL="/storage/project/${PROJECT_NAME}/shared/models/bird-mixit/bird_mixit_4sources.onnx"
DATA_DIR="/storage/project/${PROJECT_NAME}/shared/data/birdclef-2026"
SOUNDSCAPE_DIR="${DATA_DIR}/train_soundscapes"

echo "=============================================="
echo "Job: Exp 013 Phase 2 - MixIT Separation"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Mode: ${MODE}"
echo "GPU: ${USE_GPU}"
echo ""

if [[ "${USE_GPU}" == "true" ]]; then
    nvidia-smi
    echo ""
fi

# Setup venv
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
export UV_CACHE_DIR="$TMPDIR/uv-cache"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "Installing dependencies..."
if [[ "${USE_GPU}" == "true" ]]; then
    uv pip install onnxruntime-gpu soundfile resampy numpy tqdm
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

# GPU flag for python script
GPU_FLAG=""
if [[ "${USE_GPU}" == "true" ]]; then
    GPU_FLAG="--gpu"
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
            ${GPU_FLAG}
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
            ${GPU_FLAG}
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
            ${GPU_FLAG}
        ;;
    *)
        echo "Unknown mode: $MODE (expected: unlabeled, labeled, test)"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Separation complete! $(date)"
echo "=============================================="
