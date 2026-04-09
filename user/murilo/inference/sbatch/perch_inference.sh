#!/bin/bash
#SBATCH --job-name=perch-v2-tf
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=28:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Perch v2 TF inference: extract embeddings + logits from soundscapes
#
# Usage:
#   cd user/murilo/inference
#   bash sbatch/run_perch.sh --source train_soundscapes
#   bash sbatch/run_perch.sh --source all --max-files 100
#

set -eu

# Environment variables (passed from run_perch.sh or defaults)
SOURCE="${SOURCE:-train_soundscapes}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_FILES="${MAX_FILES:-}"
OVERWRITE="${OVERWRITE:-false}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"

# ============================================================================
# Determine paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    INFERENCE_DIR="$SLURM_SUBMIT_DIR"
else
    INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    INFERENCE_DIR="$(dirname "$INFERENCE_DIR")"
fi

REPO_DIR="$(cd "$INFERENCE_DIR/../../.." && pwd)"

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Perch v2 TF Inference"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Source: ${SOURCE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Project: ${PROJECT_NAME}"
echo "Inference dir: ${INFERENCE_DIR}"
echo "Repo: ${REPO_DIR}"
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

export UV_CACHE_DIR="$TMPDIR/uv-cache"

JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

# Install TensorFlow (CPU-only) + inference dependencies
echo "  Installing TensorFlow + dependencies..."
uv pip install "tensorflow-cpu>=2.15,<2.18" \
    pandas pyarrow soundfile numpy tqdm

echo "  Virtual environment ready!"
echo ""

# Verify TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
echo ""

# ============================================================================
# Environment variables
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_CPP_MIN_LOG_LEVEL=2

echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# ============================================================================
# Run inference
# ============================================================================

cd "$INFERENCE_DIR"

echo "Starting Perch v2 TF inference..."
echo ""

INFER_CMD="python perch_inference.py --source ${SOURCE} --batch-size ${BATCH_SIZE}"

if [[ -n "${MAX_FILES}" ]]; then
    INFER_CMD="${INFER_CMD} --max-files ${MAX_FILES}"
    echo "  (max files: ${MAX_FILES})"
fi
if [[ "${OVERWRITE}" == "true" ]]; then
    INFER_CMD="${INFER_CMD} --overwrite"
    echo "  (overwrite enabled)"
fi

srun ${INFER_CMD}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
