#!/bin/bash
# Launcher script for BirdSet inference jobs.
#
# Usage:
#   # Run single model on single dataset
#   bash sbatch/inference/run.sh --model birdset_convnext --dataset HSN
#
#   # Run single model on all datasets
#   bash sbatch/inference/run.sh --model birdset_convnext
#
#   # Run all models on all datasets (16 jobs)
#   bash sbatch/inference/run.sh
#
#   # Test mode: run on subset of each dataset (results saved to HSN_limit_100, etc.)
#   bash sbatch/inference/run.sh --limit 100
#
#   # Dry run (preview commands without submitting)
#   bash sbatch/inference/run.sh --dry-run
#
#   # Force re-run even if results exist
#   bash sbatch/inference/run.sh --model birdset_convnext --dataset HSN --force

set -eu

# Source common functions for paths and configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# ============================================================================
# Parse Arguments
# ============================================================================

MODEL=""
DATASET=""
LIMIT=""
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL      Model name (birdset_convnext, birdset_efficientnet_b1)"
            echo "  --dataset DATASET  Dataset code (HSN, PER, NES, UHH, POW, SSW, SNE, NBP)"
            echo "  --limit N          Run on subset of N samples (for testing end-to-end)"
            echo "  --dry-run          Preview commands without submitting jobs"
            echo "  --force            Re-run even if results already exist"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model birdset_convnext --dataset HSN"
            echo "  $0 --model birdset_convnext  # All datasets"
            echo "  $0                            # All models and datasets"
            echo "  $0 --limit 100                # Test mode on all (saves to HSN_limit_100, etc.)"
            echo "  $0 --dry-run                  # Preview all jobs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Build Job List
# ============================================================================

# Determine models to run
if [[ -n "${MODEL}" ]]; then
    MODELS_TO_RUN=("${MODEL}")
else
    MODELS_TO_RUN=("${BIRDSET_MODELS[@]}")
fi

# Determine datasets to run
if [[ -n "${DATASET}" ]]; then
    DATASETS_TO_RUN=("${DATASET}")
else
    DATASETS_TO_RUN=("${BIRDSET_DATASETS[@]}")
fi

# ============================================================================
# Helper: Get dataset path (with optional limit suffix)
# ============================================================================

get_dataset_path() {
    local dataset="$1"
    local limit="${2:-}"

    if [[ -n "${limit}" ]]; then
        echo "${dataset}_limit_${limit}"
    else
        echo "${dataset}"
    fi
}

# ============================================================================
# Submit Jobs
# ============================================================================

JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"
SUBMITTED=0
SKIPPED=0

echo "BirdSet Inference Job Launcher"
echo "=============================="
echo "Models: ${MODELS_TO_RUN[*]}"
echo "Datasets: ${DATASETS_TO_RUN[*]}"
if [[ -n "${LIMIT}" ]]; then
    echo "Limit: ${LIMIT} samples (test mode)"
fi
echo "Total combinations: $((${#MODELS_TO_RUN[@]} * ${#DATASETS_TO_RUN[@]}))"
echo ""

for model in "${MODELS_TO_RUN[@]}"; do
    for dataset in "${DATASETS_TO_RUN[@]}"; do
        # Get the dataset path (with limit suffix if applicable)
        dataset_path=$(get_dataset_path "${dataset}" "${LIMIT}")

        # Check if already complete (skip unless --force)
        if [[ "${FORCE}" == "false" ]] && check_inference_complete "${model}" "${dataset_path}"; then
            echo "[SKIP] ${model} on ${dataset_path} - results already exist"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Build export variables
        EXPORT_VARS="MODEL=${model},DATASET=${dataset}"
        if [[ -n "${LIMIT}" ]]; then
            EXPORT_VARS="${EXPORT_VARS},LIMIT=${LIMIT}"
        fi

        # Create log directory for this model/dataset
        LOG_DIR="${EXPERIMENT_DIR}/logs/inference/${model}/${dataset_path}"
        mkdir -p "${LOG_DIR}"

        # Build sbatch command with organized log output
        CMD="sbatch \
            --export=${EXPORT_VARS} \
            --output=${LOG_DIR}/slurm_%j.log \
            ${JOB_SCRIPT}"

        if [[ "${DRY_RUN}" == "true" ]]; then
            echo "[DRY-RUN] ${CMD}"
        else
            echo "[SUBMIT] ${model} on ${dataset_path}"
            ${CMD}
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

echo ""
echo "=============================="
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Submitted: ${SUBMITTED} jobs"
    echo "Skipped: ${SKIPPED} jobs (already complete)"
fi
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: ${EXPERIMENT_DIR}/logs/inference/{model}/{dataset}/"
