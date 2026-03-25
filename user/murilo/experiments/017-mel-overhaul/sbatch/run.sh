#!/bin/bash
# Launcher script for Experiment 017 mel overhaul jobs.
#
# Usage:
#   cd user/murilo/experiments/017-mel-overhaul
#
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4a.yaml
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4a.yaml --fast-dev-run
#   bash sbatch/run.sh config/efficientnet_b0_sed_v4a.yaml --gpu-check
#   bash sbatch/run.sh --all
#   bash sbatch/run.sh --all --dry-run

set -eu

DRY_RUN=false
FAST_DEV_RUN=false
GPU_CHECK=0
NUM_WORKERS=""
CONFIG=""
ALL=false
PROJECT_NAME="ps-clef2026_birdclef-0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL=true
            shift
            ;;
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
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [CONFIG_PATH | --all] [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fast-dev-run    Run 5 batches only, skip W&B"
            echo "  --gpu-check [N]   Run N epochs for GPU check (default: 1)"
            echo "  --num-workers N   Override num_workers"
            echo "  --dry-run         Preview command without submitting"
            echo ""
            echo "Configs:"
            echo "  config/efficientnet_b0_sed_v4a.yaml"
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

if [[ "${ALL}" == "true" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
    EXTRA_ARGS=()
    [[ "${DRY_RUN}" == "true" ]] && EXTRA_ARGS+=(--dry-run)
    [[ "${FAST_DEV_RUN}" == "true" ]] && EXTRA_ARGS+=(--fast-dev-run)
    [[ "${GPU_CHECK}" -gt 0 ]] && EXTRA_ARGS+=(--gpu-check "${GPU_CHECK}")
    [[ -n "${NUM_WORKERS}" ]] && EXTRA_ARGS+=(--num-workers "${NUM_WORKERS}")

    for cfg in "${EXPERIMENT_DIR}"/config/*.yaml; do
        cfg_rel="config/$(basename "$cfg")"
        echo ">>> Submitting ${cfg_rel}..."
        bash "${BASH_SOURCE[0]}" "${cfg_rel}" "${EXTRA_ARGS[@]}"
        echo ""
    done
    exit 0
fi

if [[ -z "$CONFIG" ]]; then
    echo "Error: CONFIG_PATH or --all is required"
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
