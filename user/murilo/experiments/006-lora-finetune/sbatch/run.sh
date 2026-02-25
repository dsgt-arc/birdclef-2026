#!/bin/bash
# Launcher script for Experiment 006 LoRA fine-tuning jobs.
#
# Usage:
#   cd user/murilo/experiments/006-lora-finetune
#   bash sbatch/run.sh config/eat_lora_attn.yaml
#   bash sbatch/run.sh config/eat_lora_all.yaml
#   bash sbatch/run.sh config/eat_lora_attn_mlphead.yaml
#
#   # Use adapters backend instead of peft (default)
#   bash sbatch/run.sh config/eat_lora_attn.yaml --backend adapters
#
#   # Fast dev run (5 batches, no W&B logging — useful for pipeline validation)
#   bash sbatch/run.sh config/eat_lora_attn.yaml --fast-dev-run
#
#   # Submit all 3 configs at once (peft)
#   bash sbatch/run.sh --all
#
#   # Submit all 3 configs at once (adapters)
#   bash sbatch/run.sh --all --backend adapters
#
#   # Dry run (preview command without submitting)
#   bash sbatch/run.sh config/eat_lora_attn.yaml --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
FAST_DEV_RUN=false
GPU_CHECK=0
NUM_WORKERS=""
CONFIG=""
ALL=false
BACKEND="peft"
PROJECT_NAME="ps-dsgt_clef2026-0"

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
        --backend)
            BACKEND="$2"
            if [[ "$BACKEND" != "peft" && "$BACKEND" != "adapters" ]]; then
                echo "Error: --backend must be 'peft' or 'adapters' (got: $BACKEND)"
                exit 1
            fi
            shift 2
            ;;
        --fast-dev-run)
            FAST_DEV_RUN=true
            shift
            ;;
        --gpu-check)
            GPU_CHECK="$2"
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
            echo "Usage: $0 [CONFIG_PATH | --all] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  CONFIG_PATH       Path to config file (e.g., config/eat_lora_attn.yaml)"
            echo "  --all             Submit all configs in config/ directory"
            echo ""
            echo "Options:"
            echo "  --backend B       PEFT backend: 'peft' (default) or 'adapters'"
            echo "  --project-name    PACE project allocation name (default: ps-dsgt_clef2026-0)"
            echo "  --fast-dev-run    Run 5 batches only, skip W&B (pipeline validation)"
            echo "  --gpu-check N     Run N epochs for GPU utilization check, then exit"
            echo "  --num-workers N   Override num_workers from config"
            echo "  --dry-run         Preview command without submitting job"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Available configs:"
            echo "  config/eat_lora_attn.yaml          LoRA on attention layers (primary)"
            echo "  config/eat_lora_all.yaml            LoRA on attention + MLP layers"
            echo "  config/eat_lora_attn_mlphead.yaml   LoRA attention + MLP classification head"
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

# --all: re-invoke this script for each config in config/
if [[ "${ALL}" == "true" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

    # Forward all flags except --all and CONFIG
    EXTRA_ARGS=()
    [[ "${BACKEND}" != "peft" ]] && EXTRA_ARGS+=(--backend "${BACKEND}")
    [[ "${DRY_RUN}" == "true" ]] && EXTRA_ARGS+=(--dry-run)
    [[ "${FAST_DEV_RUN}" == "true" ]] && EXTRA_ARGS+=(--fast-dev-run)
    [[ "${GPU_CHECK}" -gt 0 ]] && EXTRA_ARGS+=(--gpu-check "${GPU_CHECK}")
    [[ -n "${NUM_WORKERS}" ]] && EXTRA_ARGS+=(--num-workers "${NUM_WORKERS}")
    [[ "${PROJECT_NAME}" != "ps-dsgt_clef2026-0" ]] && EXTRA_ARGS+=(--project-name "${PROJECT_NAME}")

    for cfg in "${EXPERIMENT_DIR}"/config/*.yaml; do
        cfg_rel="config/$(basename "$cfg")"
        echo ">>> Submitting ${cfg_rel} (${BACKEND})..."
        bash "${BASH_SOURCE[0]}" "${cfg_rel}" "${EXTRA_ARGS[@]}"
        echo ""
    done
    exit 0
fi

if [[ -z "$CONFIG" ]]; then
    echo "Error: CONFIG_PATH or --all is required"
    echo "Usage: $0 [CONFIG_PATH | --all] [--backend B] [--fast-dev-run] [--dry-run]"
    echo ""
    echo "Available configs:"
    echo "  config/eat_lora_attn.yaml          LoRA on attention layers (primary)"
    echo "  config/eat_lora_all.yaml            LoRA on attention + MLP layers"
    echo "  config/eat_lora_attn_mlphead.yaml   LoRA attention + MLP classification head"
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

# Extract run name from config filename (e.g., eat_lora_attn)
RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Experiment 006: LoRA Fine-Tuning"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Backend: ${BACKEND}"
echo "Project: ${PROJECT_NAME}"
echo "Fast dev run: ${FAST_DEV_RUN}"
echo "GPU check: ${GPU_CHECK}"
echo "Num workers: ${NUM_WORKERS:-config default}"
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

# Override cpus-per-task if num-workers specified
# Max CPU:GPU ratio is 12:1 on gpu-rtx6000/gpu-v100 partitions
CPUS_OVERRIDE=""
if [[ -n "$NUM_WORKERS" ]]; then
    CPUS_OVERRIDE="--cpus-per-task=${NUM_WORKERS}"
fi

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},BACKEND=${BACKEND},FAST_DEV_RUN=${FAST_DEV_RUN},GPU_CHECK=${GPU_CHECK},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp006-${RUN_NAME} \
    ${CPUS_OVERRIDE} \
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
