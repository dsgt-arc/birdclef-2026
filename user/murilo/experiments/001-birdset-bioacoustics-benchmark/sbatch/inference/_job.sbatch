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
#
# Environment variables:
#   MODEL    - Model name: birdset_convnext, birdset_efficientnet_b1
#   DATASET  - Dataset code: HSN, PER, NES, UHH, POW, SSW, SNE, NBP
#
# Outputs saved to: artifacts/inference/{model}/{dataset}/
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

# Print job info
print_job_info "BirdSet Inference: ${MODEL} on ${DATASET}"

echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Dataset: ${DATASET}"
echo "  Batch size: ${BATCH_SIZE:-16}"
echo "  Num workers: ${NUM_WORKERS:-4}"
echo ""

# Setup environment (UV, caches, cd to repo)
echo "Setting up environment..."
setup_environment

# Start GPU monitoring
start_gpu_monitoring

# Get output directory
OUTPUT_DIR=$(get_inference_output_dir "${MODEL}" "${DATASET}")
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run inference
echo "Starting inference..."
uv run python -m birdset_benchmark.run_inference \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=============================================="
echo "Job complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
date
