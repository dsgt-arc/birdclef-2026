#!/bin/bash
# Launcher for exp 014 pseudo-labeling jobs.
#
# Usage:
#   cd user/murilo/experiments/014-pseudo-labeling
#   bash sbatch/run.sh config/direct.yaml --split labeled
#   bash sbatch/run.sh config/direct.yaml --split unlabeled
#   bash sbatch/run.sh config/direct.yaml --split unlabeled --num-workers 8
#   bash sbatch/run.sh config/direct.yaml --split labeled --max-files 5
#   bash sbatch/run.sh config/direct.yaml --split unlabeled --num-workers 8 --dry-run

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${EXPERIMENT_DIR}/logs"
JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

mkdir -p "$LOG_DIR"

# Parse args
DRY_RUN=false
CONFIG=""
SPLIT=""
MAX_FILES=""
NUM_WORKERS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --split) SPLIT="$2"; shift 2 ;;
        --max-files) MAX_FILES="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 CONFIG --split SPLIT [--num-workers N] [--max-files N] [--dry-run]"
            echo ""
            echo "  CONFIG         Path to config YAML (e.g., config/direct.yaml)"
            echo "  --split        labeled, unlabeled, or all"
            echo "  --num-workers N  Distribute across N GPUs via SLURM job array"
            echo "  --max-files N  Limit files for testing"
            echo "  --dry-run      Preview command without submitting"
            echo ""
            echo "Examples:"
            echo "  $0 config/direct.yaml --split labeled                        # 1 GPU, labeled"
            echo "  $0 config/direct.yaml --split unlabeled                      # 1 GPU, all unlabeled"
            echo "  $0 config/direct.yaml --split unlabeled --num-workers 8      # 8 GPUs in parallel"
            exit 0
            ;;
        *)
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$1"
            else
                echo "Unknown option: $1"; exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CONFIG" || -z "$SPLIT" ]]; then
    echo "Error: CONFIG and --split are required"
    echo "Usage: $0 config/direct.yaml --split labeled"
    exit 1
fi

if [[ ! -f "${EXPERIMENT_DIR}/${CONFIG}" ]]; then
    echo "Error: Config not found: ${EXPERIMENT_DIR}/${CONFIG}"
    exit 1
fi

RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Experiment 014: Pseudo-Labeling"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "Split: ${SPLIT}"
echo "Max files: ${MAX_FILES:-all}"
echo "Workers: ${NUM_WORKERS}"
echo "Logs: ${LOG_DIR}/"
echo ""

# Job array for distributed processing
ARRAY_FLAG=""
if [[ "${NUM_WORKERS}" -gt 1 ]]; then
    ARRAY_FLAG="--array=0-$((NUM_WORKERS - 1))"
fi

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_${SPLIT}_%A_%a.log \
    --export=ALL,CONFIG=${CONFIG},SPLIT=${SPLIT},MAX_FILES=${MAX_FILES},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp014-${RUN_NAME}-${SPLIT} \
    ${ARRAY_FLAG} \
    ${JOB_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    if [[ "${NUM_WORKERS}" -gt 1 ]]; then
        echo "Distributed: ${NUM_WORKERS} workers via SLURM job array"
        echo "Each worker processes ~$((10592 / NUM_WORKERS)) files"
    fi
    echo "Monitor with: squeue -u \$USER"
    echo "View logs in: ${LOG_DIR}/"
fi
