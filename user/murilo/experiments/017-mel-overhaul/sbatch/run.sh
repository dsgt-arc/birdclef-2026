#!/bin/bash
# Launcher script for Experiment 017 mel overhaul jobs.
#
# Usage:
#   cd user/murilo/experiments/017-mel-overhaul
#
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --gpu a100
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --gpu a100,h200
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --fast-dev-run
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --gpu-check
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --dry-run

set -eu

DRY_RUN=false
FAST_DEV_RUN=false
GPU_CHECK=0
NUM_WORKERS=""
CONFIG=""
PROJECT_NAME="ps-clef2026_birdclef-0"
GPU_TYPE=""

# Map short names to SLURM partition names
resolve_partitions() {
    local input="$1"
    local result=""
    IFS=',' read -ra parts <<< "$input"
    for p in "${parts[@]}"; do
        p="$(echo "$p" | tr '[:upper:]' '[:lower:]' | xargs)"
        case "$p" in
            a100|gpu-a100)       p="gpu-a100" ;;
            h100|gpu-h100)       p="gpu-h100" ;;
            h200|gpu-h200)       p="gpu-h200" ;;
            rtx6000|rtx|gpu-rtx6000) p="gpu-rtx6000" ;;
            v100|gpu-v100)       p="gpu-v100" ;;
            *)
                echo "Error: unknown GPU type '$p'. Options: a100, h100, h200, rtx6000, v100" >&2
                exit 1
                ;;
        esac
        result="${result:+$result,}$p"
    done
    echo "$result"
}

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
            if [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]]; then
                GPU_CHECK="$2"
                shift 2
            else
                GPU_CHECK=1
                shift
            fi
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 CONFIG_PATH [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu TYPE        GPU partition: a100, h100, h200, rtx6000, v100"
            echo "                    Comma-separated for fallback: --gpu a100,rtx6000"
            echo "                    Default: gpu-rtx6000,gpu-v100"
            echo "  --fast-dev-run    Run 5 batches only, skip W&B"
            echo "  --gpu-check [N]   Run N epochs for GPU check (default: 1)"
            echo "  --num-workers N   Override num_workers"
            echo "  --dry-run         Preview command without submitting"
            echo ""
            echo "Configs:"
            ls config/*.yaml 2>/dev/null | while read -r cfg; do echo "  $cfg"; done
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

# Resolve GPU partition
if [[ -n "$GPU_TYPE" ]]; then
    PARTITION="$(resolve_partitions "$GPU_TYPE")"
else
    PARTITION="gpu-rtx6000,gpu-v100"
fi

if [[ -z "$CONFIG" ]]; then
    echo "Error: CONFIG_PATH is required"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Experiment 017: Mel Overhaul"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Partition: ${PARTITION}"
echo ""

if [[ ! -f "${EXPERIMENT_DIR}/${CONFIG}" ]]; then
    echo "Error: Config not found: ${EXPERIMENT_DIR}/${CONFIG}"
    exit 1
fi

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

CPUS_OVERRIDE=""
if [[ -n "$NUM_WORKERS" ]]; then
    CPUS_OVERRIDE="--cpus-per-task=${NUM_WORKERS}"
fi

CMD="sbatch \
    --partition=${PARTITION} \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},GPU_CHECK=${GPU_CHECK},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp017-${RUN_NAME} \
    ${CPUS_OVERRIDE} \
    ${JOB_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs: ${LOG_DIR}/"
fi
