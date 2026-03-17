#!/bin/bash
# Launcher script for Experiment 010 submission pipeline jobs.
#
# Model naming: {encoder}-{year}[-{variant}]
#   encoder: eat, convnext, protoclr
#   year:    2025 (exps 006/007/009), 2026 (exp 011)
#   variant: (none) = 80/10/10 split, fulldata = 100% focal
#
# Usage:
#   cd user/murilo/experiments/010-submission-pipeline
#
#   # ONNX export (GPU)
#   bash sbatch/run.sh export --model protoclr-2025          # 2025-trained (exp 009)
#   bash sbatch/run.sh export --model protoclr-2026          # 2026 80/10/10 (exp 011)
#   bash sbatch/run.sh export --model protoclr-2026-fulldata # 2026 fulldata (exp 011 --train-all)
#   bash sbatch/run.sh export --model all-2025               # all 3 encoders, 2025
#   bash sbatch/run.sh export --model all-2026               # all 3 encoders, 2026 80/10/10
#   bash sbatch/run.sh export --model all-2026-fulldata      # all 3 encoders, 2026 fulldata
#   bash sbatch/run.sh export --model all-2026-fulldata --dry-run
#
#   # Local validation (CPU)
#   bash sbatch/run.sh validate --model protoclr-2026-fulldata
#   bash sbatch/run.sh validate --model all-2026

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
    echo "  --model M   Model to process ({encoder}-{year}[-{variant}]):"
    echo "              eat-2025, convnext-2025, protoclr-2025, all-2025"
    echo "              eat-2026, convnext-2026, protoclr-2026, all-2026"
    echo "              eat-2026-fulldata, convnext-2026-fulldata, protoclr-2026-fulldata, all-2026-fulldata"
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
