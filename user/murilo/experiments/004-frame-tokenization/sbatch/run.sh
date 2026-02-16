#!/bin/bash
# Launcher script for Experiment 004 training jobs.
#
# Usage:
#   cd user/murilo/experiments/004-frame-tokenization
#   bash sbatch/run.sh config/vertical_128x2.yaml
#   bash sbatch/run.sh config/baseline_16x16.yaml
#   bash sbatch/run.sh config/vertical_128x4.yaml
#   bash sbatch/run.sh config/vertical_128x1.yaml
#
#   # Run all 4 configurations
#   bash sbatch/run.sh --all
#
#   # Fast dev run (5 batches, no W&B logging — pipeline validation)
#   bash sbatch/run.sh config/vertical_128x2.yaml --fast-dev-run
#
#   # Dry run (preview command without submitting)
#   bash sbatch/run.sh config/vertical_128x2.yaml --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
FAST_DEV_RUN=false
RUN_ALL=false
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
        --all)
            RUN_ALL=true
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
            echo "  CONFIG_PATH       Path to config file (e.g., config/vertical_128x2.yaml)"
            echo ""
            echo "Options:"
            echo "  --all             Submit all 4 configurations"
            echo "  --project-name    PACE project allocation name (default: ps-dsgt_clef2026-0)"
            echo "  --fast-dev-run    Run 5 batches only, skip W&B (pipeline validation)"
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

# ============================================================================
# Determine paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Submit function
# ============================================================================

submit_job() {
    local config="$1"
    local run_name
    run_name="$(basename "$config" .yaml)"

    echo "=============================================="
    echo "Experiment 004: Frame-Based Tokenization"
    echo "=============================================="
    echo "Experiment dir: ${EXPERIMENT_DIR}"
    echo "Config: ${config}"
    echo "Run name: ${run_name}"
    echo "Project: ${PROJECT_NAME}"
    echo "Fast dev run: ${FAST_DEV_RUN}"
    echo ""

    # Verify config exists
    if [[ ! -f "${EXPERIMENT_DIR}/${config}" ]]; then
        echo "Error: Config file not found: ${EXPERIMENT_DIR}/${config}"
        return 1
    fi

    # Create log directory
    LOG_DIR="${EXPERIMENT_DIR}/logs"
    mkdir -p "${LOG_DIR}"

    # Job script
    JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

    CMD="sbatch \
        --output=${LOG_DIR}/slurm_${run_name}_%j.log \
        --export=ALL,CONFIG=${config},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN} \
        --job-name=exp004-${run_name} \
        ${JOB_SCRIPT}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY-RUN] ${CMD}"
    else
        echo "Submitting job..."
        ${CMD}
        echo ""
        echo "Job submitted!"
    fi
    echo ""
}

# ============================================================================
# Submit job(s)
# ============================================================================

if [[ "${RUN_ALL}" == "true" ]]; then
    echo "Submitting all 4 configurations..."
    echo ""
    submit_job "config/baseline_16x16.yaml"
    submit_job "config/vertical_128x2.yaml"
    submit_job "config/vertical_128x4.yaml"
    submit_job "config/vertical_128x1.yaml"
    echo "=============================================="
    echo "All jobs submitted! Monitor with: squeue -u \$USER"
    echo "=============================================="
elif [[ -z "$CONFIG" ]]; then
    echo "Error: CONFIG_PATH is required (or use --all)"
    echo "Usage: $0 CONFIG_PATH [--project-name NAME] [--fast-dev-run] [--dry-run]"
    echo ""
    echo "Available configs:"
    echo "  config/baseline_16x16.yaml    (16x16 scratch baseline)"
    echo "  config/vertical_128x2.yaml    (Config A: primary test)"
    echo "  config/vertical_128x4.yaml    (Config B: coarser temporal)"
    echo "  config/vertical_128x1.yaml    (Config C: finest temporal)"
    echo ""
    echo "Or submit all: $0 --all"
    exit 1
else
    submit_job "$CONFIG"
    echo "=============================================="
    echo "Monitor with: squeue -u \$USER"
    echo "View logs in: ${EXPERIMENT_DIR}/logs/"
    echo "=============================================="
fi
