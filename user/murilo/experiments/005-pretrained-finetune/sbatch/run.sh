#!/bin/bash
# Launcher script for Experiment 005 fine-tuning jobs.
#
# Usage:
#   cd user/murilo/experiments/005-pretrained-finetune
#   bash sbatch/run.sh config/eat_frozen.yaml
#   bash sbatch/run.sh config/eat_finetune.yaml
#
#   # Fast dev run (5 batches, no W&B logging — useful for pipeline validation)
#   bash sbatch/run.sh config/eat_frozen.yaml --fast-dev-run
#
#   # Dry run (preview command without submitting)
#   bash sbatch/run.sh config/eat_frozen.yaml --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
FAST_DEV_RUN=false
GPU_CHECK=0
CONFIG=""
PROJECT_NAME="ps-dsgt_clef2026-0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --fast-dev-run)
            FAST_DEV_RUN=true
            shift
            ;;
        --gpu-check)
            GPU_CHECK="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 CONFIG_PATH [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  CONFIG_PATH       Path to config file (e.g., config/eat_frozen.yaml)"
            echo ""
            echo "Options:"
            echo "  --project-name    PACE project allocation name (default: ps-dsgt_clef2026-0)"
            echo "  --fast-dev-run    Run 5 batches only, skip W&B (pipeline validation)"
            echo "  --gpu-check N     Run N epochs for GPU utilization check, then exit"
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
    echo "Usage: $0 CONFIG_PATH [--project-name NAME] [--fast-dev-run] [--dry-run]"
    echo ""
    echo "Available configs:"
    echo "  config/eat_frozen.yaml      EAT linear probe (frozen encoder)"
    echo "  config/eat_finetune.yaml    EAT full fine-tune (staged unfreezing)"
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

# Extract run name from config filename (e.g., eat_frozen)
RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Experiment 005: Pretrained Fine-tuning"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Project: ${PROJECT_NAME}"
echo "Fast dev run: ${FAST_DEV_RUN}"
echo "GPU check: ${GPU_CHECK}"
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
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},GPU_CHECK=${GPU_CHECK} \
    --job-name=exp005-${RUN_NAME} \
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
