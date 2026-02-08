#!/bin/bash
# Launcher script for Experiment 003 training jobs.
#
# Usage:
#   cd user/murilo/experiments/003-model-training
#   bash sbatch/run.sh config/convnext_tiny_scratch.yaml
#   bash sbatch/run.sh config/vit_small_scratch.yaml
#   bash sbatch/run.sh config/convnext_tiny_pretrained.yaml
#   bash sbatch/run.sh config/vit_small_pretrained.yaml
#
#   # With a different PACE project allocation
#   bash sbatch/run.sh config/convnext_tiny_scratch.yaml --project-name my-other-project
#
#   # Dry run (preview command without submitting)
#   bash sbatch/run.sh config/convnext_tiny_scratch.yaml --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
CONFIG=""
PROJECT_NAME="ps-dsgt_clef2026-0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 CONFIG_PATH [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  CONFIG_PATH       Path to config file (e.g., config/convnext_tiny_scratch.yaml)"
            echo ""
            echo "Options:"
            echo "  --project-name    PACE project allocation name (default: ps-dsgt_clef2026-0)"
            echo "  --dry-run         Preview command without submitting job"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: CONFIG_PATH is required"
    echo "Usage: $0 CONFIG_PATH [--project-name NAME] [--dry-run]"
    echo ""
    echo "Available configs:"
    echo "  config/convnext_tiny_scratch.yaml"
    echo "  config/vit_small_scratch.yaml"
    echo "  config/convnext_tiny_pretrained.yaml"
    echo "  config/vit_small_pretrained.yaml"
    exit 1
fi

# ============================================================================
# Determine paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Setup
# ============================================================================

# Extract run name from config filename (e.g., convnext_tiny_scratch)
RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Experiment 003: Model Training"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Project: ${PROJECT_NAME}"
echo ""

# Verify config exists
if [[ ! -f "${EXPERIMENT_DIR}/${CONFIG}" ]]; then
    echo "Error: Config file not found: ${EXPERIMENT_DIR}/${CONFIG}"
    exit 1
fi

# Create log directory
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Job script
JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

# ============================================================================
# Submit job
# ============================================================================

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME} \
    --job-name=exp003-${RUN_NAME} \
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
