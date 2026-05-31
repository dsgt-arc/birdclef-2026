#!/bin/bash
# Launcher for Exp 038 Stage 1 — XC supervised pretraining. Run with `bash` (it always sbatches
# pretrain.sh as a detached GPU job — same launcher/worker pattern as exp 037 run.sh/_job.sh).
#
# Usage:
#   cd user/murilo/experiments/038-xc-pretraining
#   bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml                  # primary
#   bash sbatch/run.sh configs/pretrain_cellB_focalbce.yaml                 # insurance arm
#   bash sbatch/run.sh configs/pretrain_cellC_nomixup.yaml                  # insurance arm
#   bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml --fast-dev-run   # gradient-flow check
#   bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml --gpu a100,rtx6000
#   bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml --dry-run        # preview, no submit
set -eu

DRY_RUN=false
FAST_DEV_RUN=false
NUM_WORKERS=""
CONFIG=""
PROJECT_NAME="ps-clef2026_birdclef-0"
GPU_TYPE=""

resolve_partitions() {
    local result=""
    IFS=',' read -ra parts <<< "$1"
    for p in "${parts[@]}"; do
        p="$(echo "$p" | tr '[:upper:]' '[:lower:]' | xargs)"
        case "$p" in
            a100|gpu-a100)           p="gpu-a100" ;;
            h100|gpu-h100)           p="gpu-h100" ;;
            h200|gpu-h200)           p="gpu-h200" ;;
            rtx6000|rtx|gpu-rtx6000) p="gpu-rtx6000" ;;
            v100|gpu-v100)           p="gpu-v100" ;;
            *) echo "Error: unknown GPU type '$p' (a100|h100|h200|rtx6000|v100)" >&2; exit 1 ;;
        esac
        result="${result:+$result,}$p"
    done
    echo "$result"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)      DRY_RUN=true; shift ;;
        --fast-dev-run) FAST_DEV_RUN=true; shift ;;
        --gpu)          GPU_TYPE="$2"; shift 2 ;;
        --num-workers)  NUM_WORKERS="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 CONFIG_PATH [--gpu TYPE] [--num-workers N] [--fast-dev-run] [--dry-run]"
            ls configs/*.yaml 2>/dev/null | grep -v _common | sed 's/^/  /'
            exit 0 ;;
        *)
            if [[ -z "$CONFIG" ]]; then CONFIG="$1"; else echo "Unknown arg: $1" >&2; exit 1; fi
            shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then echo "Error: CONFIG_PATH required" >&2; exit 1; fi

# Default partition: A100 (EffNetV2-S at batch 128 + 256-mel wants the memory). Override with --gpu.
if [[ -n "$GPU_TYPE" ]]; then
    PARTITION="$(resolve_partitions "$GPU_TYPE")"
else
    PARTITION="gpu-a100"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="$(basename "$CONFIG" .yaml)"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${EXPERIMENT_DIR}/${CONFIG}" ]]; then
    echo "Error: config not found: ${EXPERIMENT_DIR}/${CONFIG}" >&2; exit 1
fi

echo "=============================================="
echo "Exp 038 Stage 1 — XC supervised pretraining"
echo "=============================================="
echo "Config:    ${CONFIG}"
echo "Partition: ${PARTITION}"
echo "Fast dev:  ${FAST_DEV_RUN}"
echo ""

CMD="sbatch \
    --partition=${PARTITION} \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp038-${RUN_NAME} \
    ${SCRIPT_DIR}/pretrain.sh"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    ${LOG_DIR}/"
fi
