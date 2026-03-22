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
#
# Experiment 013, Phase 2: Run Bird MixIT separation on soundscapes
#
# Usage:
#   cd user/murilo/experiments/013-mixit-pseudo-labeling
#
#   # Separate all unlabeled soundscapes
#   sbatch sbatch/separate.sh
#
#   # Separate labeled soundscapes only (for validation)
#   MODE=labeled sbatch sbatch/separate.sh
#
#   # Quick test (10 files)
#   MODE=test sbatch sbatch/separate.sh

set -eu

MODE="${MODE:-unlabeled}"

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
echo ""

# Setup venv
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
export UV_CACHE_DIR="$TMPDIR/uv-cache"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "Installing dependencies..."
uv pip install onnxruntime soundfile resampy numpy tqdm
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs artifacts/separated artifacts/separated_labeled

# Verify model exists
if [ ! -f "$MIXIT_MODEL" ]; then
    echo "ERROR: ONNX model not found at $MIXIT_MODEL"
    echo "Run Phase 1 first: sbatch sbatch/convert.sh"
    exit 1
fi

# Run separation based on mode
case "$MODE" in
    unlabeled)
        echo "Separating unlabeled soundscapes..."
        python scripts/separate_soundscapes.py \
            --model "$MIXIT_MODEL" \
            --soundscape-dir "$SOUNDSCAPE_DIR" \
            --data-dir "$DATA_DIR" \
            --output-dir artifacts/separated \
            --exclude-labeled \
            --num-threads "$SLURM_CPUS_PER_TASK"
        ;;
    labeled)
        echo "Separating labeled soundscapes (for validation)..."
        python scripts/separate_soundscapes.py \
            --model "$MIXIT_MODEL" \
            --soundscape-dir "$SOUNDSCAPE_DIR" \
            --data-dir "$DATA_DIR" \
            --output-dir artifacts/separated_labeled \
            --only-labeled \
            --num-threads "$SLURM_CPUS_PER_TASK"
        ;;
    test)
        echo "Test run (10 files)..."
        python scripts/separate_soundscapes.py \
            --model "$MIXIT_MODEL" \
            --soundscape-dir "$SOUNDSCAPE_DIR" \
            --data-dir "$DATA_DIR" \
            --output-dir artifacts/separated_test \
            --max-files 10 \
            --num-threads "$SLURM_CPUS_PER_TASK"
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
