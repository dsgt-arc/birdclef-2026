#!/bin/bash
# Launcher script for Experiment 002 training job.
#
# Usage:
#   cd user/murilo/experiments/002-supervised-vit-baseline
#   bash sbatch/run.sh
#
#   # Dry run (preview command without submitting)
#   bash sbatch/run.sh --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Preview command without submitting job"
            echo "  -h, --help   Show this help message"
            exit 0
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

# ============================================================================
# Setup
# ============================================================================

echo "=============================================="
echo "Experiment 002: Supervised ViT Baseline"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo ""

# Create log directory
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Job script
JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

# ============================================================================
# Submit job
# ============================================================================

CMD="sbatch \
    --output=${LOG_DIR}/slurm_%j.log \
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
