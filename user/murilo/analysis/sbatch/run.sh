#!/bin/bash
# Launcher for pre-computing per-sample losses.
#
# Usage:
#   cd user/murilo/analysis
#
#   # Default: v5j model
#   bash sbatch/run.sh
#
#   # Custom model:
#   bash sbatch/run.sh --config path/to/config.yaml --checkpoint path/to/ckpt
#
#   # Options:
#   bash sbatch/run.sh --gpu a100
#   bash sbatch/run.sh --output losses_v6l.parquet
#   bash sbatch/run.sh --dry-run

set -eu

DRY_RUN=false
PROJECT_NAME="ps-clef2026_birdclef-0"
GPU_TYPE=""
BATCH_SIZE=64

# Default: v5j model
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
USER_DIR="$(dirname "$ANALYSIS_DIR")"

CONFIG="${USER_DIR}/experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml"
CHECKPOINT="/storage/project/${PROJECT_NAME}/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt"
OUTPUT="losses_v5j.parquet"

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
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       YAML config (default: v5j)"
            echo "  --checkpoint PATH   Checkpoint path (default: v5j)"
            echo "  --output FILE       Output parquet filename (default: losses_v5j.parquet)"
            echo "  --gpu TYPE          GPU partition: a100, h100, rtx6000, v100 (default: rtx6000,v100)"
            echo "  --batch-size N      Batch size (default: 64)"
            echo "  --dry-run           Preview command without submitting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve GPU partition
if [[ -n "$GPU_TYPE" ]]; then
    PARTITION="$(resolve_partitions "$GPU_TYPE")"
else
    PARTITION="gpu-rtx6000,gpu-v100"
fi

echo "=============================================="
echo "Pre-compute Losses for Model Explorer"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "Checkpoint: $(basename "${CHECKPOINT}")"
echo "Output: ${OUTPUT}"
echo "Partition: ${PARTITION}"
echo ""

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

LOG_DIR="${ANALYSIS_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"
RUN_NAME="$(basename "$OUTPUT" .parquet)"

CMD="sbatch \
    --partition=${PARTITION} \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,CONFIG=${CONFIG},CHECKPOINT=${CHECKPOINT},OUTPUT=${OUTPUT},BATCH_SIZE=${BATCH_SIZE},PROJECT_NAME=${PROJECT_NAME} \
    --job-name=losses-${RUN_NAME} \
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
