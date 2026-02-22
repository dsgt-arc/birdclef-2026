#!/bin/bash
#SBATCH --job-name=birdset-infer
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --output=logs/inference-%j.log
#
# BirdSet Direct Inference Job
#
# Runs inference with BirdSet-native models (ConvNeXT, EfficientNetB1) on BirdSet datasets.
# These models were trained on BirdSet and can directly predict species without additional training.
#
# Usage:
#   sbatch --export=MODEL=birdset_convnext,DATASET=HSN sbatch/inference/_job.sh
#   sbatch --export=MODEL=birdset_convnext,DATASET=HSN,LIMIT=100 sbatch/inference/_job.sh
#
# Environment variables:
#   MODEL    - Model name: birdset_convnext, birdset_efficientnet_b1
#   DATASET  - Dataset code: HSN, PER, NES, UHH, POW, SSW, SNE, NBP
#   LIMIT    - (Optional) Number of samples to use (for testing)
#
# Outputs saved to: artifacts/inference/{model}/{dataset}/ (or {dataset}_limit_{N}/ if LIMIT is set)
#   - metrics.json: cmAP, F1, Top-K accuracy, throughput
#   - predictions.npz: y_pred and y_true arrays

set -eu

# ============================================================================
# Determine paths
# ============================================================================

# SLURM copies script to temp location, so use SLURM_SUBMIT_DIR if available
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
fi

USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"
ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts"

# ============================================================================
# Validate required environment variables
# ============================================================================

if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL environment variable not set"
    echo "Usage: sbatch --export=MODEL=birdset_convnext,DATASET=HSN _job.sh"
    exit 1
fi

if [[ -z "${DATASET:-}" ]]; then
    echo "ERROR: DATASET environment variable not set"
    echo "Usage: sbatch --export=MODEL=birdset_convnext,DATASET=HSN _job.sh"
    exit 1
fi

# Construct dataset path (with optional limit suffix)
if [[ -n "${LIMIT:-}" ]]; then
    DATASET_PATH="${DATASET}_limit_${LIMIT}"
else
    DATASET_PATH="${DATASET}"
fi

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: BirdSet Inference: ${MODEL} on ${DATASET_PATH}"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${REPO_DIR}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo ""
nvidia-smi
echo ""

echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Dataset: ${DATASET}"
if [[ -n "${LIMIT:-}" ]]; then
    echo "  Limit: ${LIMIT} samples"
fi
echo "  Output path: ${DATASET_PATH}"
echo "  Batch size: ${BATCH_SIZE:-16}"
echo "  Num workers: ${NUM_WORKERS:-4}"
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

# Create job-specific venv in TMPDIR for fast local I/O
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV"

# Activate the virtual environment
source "$JOB_VENV/bin/activate"

# Install the user package (includes all dependencies from pyproject.toml)
echo "  Installing dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

# ============================================================================
# Environment variables
# ============================================================================

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONWARNINGS=ignore

# ============================================================================
# GPU Monitoring
# ============================================================================

# Store GPU stats in same directory structure as logs
GPU_LOG_DIR="${EXPERIMENT_DIR}/logs/inference/${MODEL}/${DATASET_PATH}"
mkdir -p "${GPU_LOG_DIR}"
GPU_STATS_FILE="${GPU_LOG_DIR}/gpu_stats_${SLURM_JOB_ID:-local}.csv"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > "$GPU_STATS_FILE" &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

echo "GPU monitoring started (log: $GPU_STATS_FILE)"
echo ""

# ============================================================================
# Run inference
# ============================================================================

OUTPUT_DIR="${ARTIFACTS_DIR}/inference/${MODEL}/${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Build inference command
INFERENCE_CMD="python -m birdclef.run_inference \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --batch-size ${BATCH_SIZE:-16} \
    --num-workers ${NUM_WORKERS:-4} \
    --output-dir ${OUTPUT_DIR}"

# Add subset-size flag if LIMIT is set
if [[ -n "${LIMIT:-}" ]]; then
    INFERENCE_CMD="${INFERENCE_CMD} --subset-size ${LIMIT}"
fi

echo "Starting inference..."
eval "${INFERENCE_CMD}"

echo ""
echo "=============================================="
echo "Job complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
date
