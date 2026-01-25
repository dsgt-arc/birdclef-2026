#!/bin/bash
#SBATCH --job-name=birdset-infer
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t240 -qembers
#SBATCH --output=logs/inference-%j.out
#
# BirdSet Direct Inference Job
#
# Runs inference with BirdSet-native models (ConvNeXT, EfficientNetB1) on BirdSet datasets.
# These models were trained on BirdSet and can directly predict species without additional training.
#
# Usage:
#   sbatch --export=MODEL=birdset_convnext,DATASET=HSN sbatch/inference/_job.sbatch
#   sbatch --export=MODEL=birdset_convnext,DATASET=HSN,LIMIT=100 sbatch/inference/_job.sbatch
#
# Environment variables:
#   MODEL    - Model name: birdset_convnext, birdset_efficientnet_b1
#   DATASET  - Dataset code: HSN, PER, NES, UHH, POW, SSW, SNE, NBP
#   LIMIT    - (Optional) Number of samples to use (for testing)
#
# Outputs saved to: artifacts/inference/{model}/{dataset}/ (or {dataset}_limit_{N}/ if LIMIT is set)
#   - metrics.json: cmAP, F1, Top-K accuracy, throughput
#   - predictions.npz: y_pred and y_true arrays

# Source common functions (sets up paths, environment)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Validate required environment variables
if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL environment variable not set"
    echo "Usage: sbatch --export=MODEL=birdset_convnext,DATASET=HSN _job.sbatch"
    exit 1
fi

if [[ -z "${DATASET:-}" ]]; then
    echo "ERROR: DATASET environment variable not set"
    echo "Usage: sbatch --export=MODEL=birdset_convnext,DATASET=HSN _job.sbatch"
    exit 1
fi

# Construct dataset path (with optional limit suffix)
if [[ -n "${LIMIT:-}" ]]; then
    DATASET_PATH="${DATASET}_limit_${LIMIT}"
else
    DATASET_PATH="${DATASET}"
fi

# Print job info
print_job_info "BirdSet Inference: ${MODEL} on ${DATASET_PATH}"

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

# Setup environment (UV, caches, cd to repo)
echo "Setting up environment..."
setup_environment

# Start GPU monitoring
start_gpu_monitoring

# Get output directory (use DATASET_PATH which includes limit suffix)
OUTPUT_DIR=$(get_inference_output_dir "${MODEL}" "${DATASET_PATH}")
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Build inference command
INFERENCE_CMD="uv run python -m birdset_benchmark.run_inference \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --batch-size ${BATCH_SIZE:-16} \
    --num-workers ${NUM_WORKERS:-4} \
    --output-dir ${OUTPUT_DIR}"

# Add subset-size flag if LIMIT is set
if [[ -n "${LIMIT:-}" ]]; then
    INFERENCE_CMD="${INFERENCE_CMD} --subset-size ${LIMIT}"
fi

# Run inference
echo "Starting inference..."
eval "${INFERENCE_CMD}"

echo ""
echo "=============================================="
echo "Job complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
date
