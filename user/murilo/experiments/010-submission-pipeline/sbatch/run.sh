#!/bin/bash
# Launcher script for Experiment 010 submission pipeline jobs.
#
# Usage:
#   cd user/murilo/experiments/010-submission-pipeline
#
#   # ONNX export (GPU)
#   bash sbatch/run.sh export                  # export all 3 models
#   bash sbatch/run.sh export --model eat      # export specific model
#   bash sbatch/run.sh export --dry-run        # preview without submitting
#
#   # Local validation (CPU)
#   bash sbatch/run.sh validate                # validate all models
#   bash sbatch/run.sh validate --model eat    # validate specific model

set -eu

# ============================================================================
# Parse arguments
# ============================================================================

JOB_TYPE=""
MODEL="all"
DRY_RUN=false

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <export|validate> [--model MODEL] [--dry-run]"
    echo ""
    echo "Jobs:"
    echo "  export     GPU job: ONNX export (builds models, merges LoRA, traces)"
    echo "  validate   CPU job: timing test on train_soundscapes"
    echo ""
    echo "Options:"
    echo "  --model M   Model to process: eat, convnext, protoclr, all (default: all)"
    echo "  --dry-run   Preview sbatch command without submitting"
    exit 1
fi

JOB_TYPE="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Determine paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

# Create log directory (must exist before sbatch resolves --output path)
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Experiment 010: Submission Pipeline"
echo "=============================================="
echo "Job type: ${JOB_TYPE}"
echo "Model: ${MODEL}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""

# ============================================================================
# Submit job
# ============================================================================

case "$JOB_TYPE" in
    export)
        JOB_SCRIPT="${SCRIPT_DIR}/export.sbatch"
        JOB_NAME="exp010-export-${MODEL}"
        LOG_FILE="${LOG_DIR}/slurm_export_${MODEL}_%j.log"
        ;;
    validate)
        JOB_SCRIPT="${SCRIPT_DIR}/validate.sbatch"
        JOB_NAME="exp010-validate-${MODEL}"
        LOG_FILE="${LOG_DIR}/slurm_validate_${MODEL}_%j.log"
        ;;
    *)
        echo "Unknown job type: $JOB_TYPE (must be 'export' or 'validate')"
        exit 1
        ;;
esac

CMD="sbatch \
    --output=${LOG_FILE} \
    --export=ALL,MODEL=${MODEL} \
    --job-name=${JOB_NAME} \
    ${JOB_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    echo "=============================================="
    echo "Job submitted!"
    echo "=============================================="
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "View logs in: ${LOG_DIR}/"
fi
