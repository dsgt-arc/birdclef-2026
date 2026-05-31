#!/bin/bash
# Launcher for Exp 038 Stage 2 — fine-tune an XC-pretrained EffNetV2-S backbone on the 234 targets.
# Run with `bash` (it sbatches finetune.sh as a detached GPU job — same launcher/worker pattern as
# run.sh/pretrain.sh).
#
# Usage:
#   cd user/murilo/experiments/038-xc-pretraining
#   bash sbatch/run_finetune.sh configs/finetune_cellA.yaml --fold 0            # one fold
#   bash sbatch/run_finetune.sh configs/finetune_cellA.yaml --fold 0 --fast-dev-run
#   bash sbatch/run_finetune.sh configs/finetune_cellC.yaml --fold 2 --gpu a100,rtx6000
#   # all 5 folds for both cells (compute is free; LB-submit single-fold first):
#   for f in 0 1 2 3 4; do bash sbatch/run_finetune.sh configs/finetune_cellA.yaml --fold $f; done
#   for f in 0 1 2 3 4; do bash sbatch/run_finetune.sh configs/finetune_cellC.yaml --fold $f; done
set -eu

DRY_RUN=false
FAST_DEV_RUN=false
NUM_WORKERS=""
CONFIG=""
FOLD=""
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
        --fold)         FOLD="$2"; shift 2 ;;
        --gpu)          GPU_TYPE="$2"; shift 2 ;;
        --num-workers)  NUM_WORKERS="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 CONFIG_PATH --fold N [--gpu TYPE] [--num-workers N] [--fast-dev-run] [--dry-run]"
            ls configs/finetune_*.yaml 2>/dev/null | sed 's/^/  /'
            exit 0 ;;
        *)
            if [[ -z "$CONFIG" ]]; then CONFIG="$1"; else echo "Unknown arg: $1" >&2; exit 1; fi
            shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then echo "Error: CONFIG_PATH required" >&2; exit 1; fi

# Default partition: rtx6000 (the reliable parallel option for the 10-fold A/C grid; 24 GB fits
# batch 32 EffNetV2-S + the GPU Perch KD teacher). Override with --gpu a100,rtx6000 etc.
if [[ -n "$GPU_TYPE" ]]; then
    PARTITION="$(resolve_partitions "$GPU_TYPE")"
else
    PARTITION="gpu-rtx6000"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="$(basename "$CONFIG" .yaml)"
LOG_TAG="$RUN_NAME"
if [[ -n "$FOLD" ]]; then LOG_TAG="${RUN_NAME}_fold${FOLD}"; fi
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${EXPERIMENT_DIR}/${CONFIG}" ]]; then
    echo "Error: config not found: ${EXPERIMENT_DIR}/${CONFIG}" >&2; exit 1
fi

echo "=============================================="
echo "Exp 038 Stage 2 — XC-backbone fine-tune (KD-SED)"
echo "=============================================="
echo "Config:    ${CONFIG}"
echo "Fold:      ${FOLD:-<none>}"
echo "Partition: ${PARTITION}"
echo "Fast dev:  ${FAST_DEV_RUN}"
echo ""

CMD="sbatch \
    --partition=${PARTITION} \
    --output=${LOG_DIR}/slurm_${LOG_TAG}_%j.log \
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},FOLD=${FOLD},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp038-ft-${LOG_TAG} \
    ${SCRIPT_DIR}/finetune.sh"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    ${LOG_DIR}/"
fi
