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
            echo "  --dry-run          Preview commands without submitting jobs"
            echo "  --force            Re-run even if results already exist"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model birdset_convnext --dataset HSN"
            echo "  $0 --model birdset_convnext  # All datasets"
            echo "  $0                            # All models and datasets"
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
# Submit Jobs
# ============================================================================

JOB_SCRIPT="${SCRIPT_DIR}/_job.sbatch"
SUBMITTED=0
SKIPPED=0

echo "BirdSet Inference Job Launcher"
echo "=============================="
echo "Models: ${MODELS_TO_RUN[*]}"
echo "Datasets: ${DATASETS_TO_RUN[*]}"
echo "Total combinations: $((${#MODELS_TO_RUN[@]} * ${#DATASETS_TO_RUN[@]}))"
echo ""

# Ensure logs directory exists
mkdir -p "${EXPERIMENT_DIR}/logs"

for model in "${MODELS_TO_RUN[@]}"; do
    for dataset in "${DATASETS_TO_RUN[@]}"; do
        # Check if already complete (skip unless --force)
        if [[ "${FORCE}" == "false" ]] && check_inference_complete "${model}" "${dataset}"; then
            echo "[SKIP] ${model} on ${dataset} - results already exist"
            ((SKIPPED++))
            continue
        fi

        # Build sbatch command
        CMD="sbatch --export=MODEL=${model},DATASET=${dataset} ${JOB_SCRIPT}"

        if [[ "${DRY_RUN}" == "true" ]]; then
            echo "[DRY-RUN] ${CMD}"
        else
            echo "[SUBMIT] ${model} on ${dataset}"
            # Change to experiment directory so logs go to the right place
            (cd "${EXPERIMENT_DIR}" && ${CMD})
            ((SUBMITTED++))
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
echo "View logs in: ${EXPERIMENT_DIR}/logs/"
