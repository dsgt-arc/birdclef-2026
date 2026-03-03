#!/bin/bash
# Launcher script for Experiment 008 frame tokenization jobs.
#
# Usage:
#   cd user/murilo/experiments/008-frame-tokenization-eat
#   bash sbatch/run.sh config/config_a_baseline.yaml
#   bash sbatch/run.sh config/config_c1a_frame_2x128.yaml
#
#   # Fast dev run (5 batches, no W&B logging — useful for pipeline validation)
#   bash sbatch/run.sh config/config_a_baseline.yaml --fast-dev-run
#
#   # Submit all configs at once
#   bash sbatch/run.sh --all
#   bash sbatch/run.sh --all --dry-run
#
#   # Dry run (preview command without submitting job)
#   bash sbatch/run.sh config/config_a_baseline.yaml --dry-run
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
            echo "  CONFIG_PATH       Path to config file (e.g., config/config_a_baseline.yaml)"
            echo "  --all             Submit all configs in config/ directory"
            echo ""
            echo "Options:"
            echo "  --project-name    PACE project allocation name (default: ps-clef2026_birdclef-0)"
            echo "  --fast-dev-run    Run 5 batches only, skip W&B (pipeline validation)"
            echo "  --gpu-check N     Run N epochs for GPU utilization check, then exit"
            echo "  --num-workers N   Override num_workers from config"
            echo "  --dry-run         Preview command without submitting job"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Available configs:"
            echo "  config/config_a_baseline.yaml            Pretrained 16x16 (frozen) + LoRA"
            echo "  config/config_b_control.yaml             Random 16x16 (trainable) + LoRA"
            echo "  config/config_c1a_frame_2x128.yaml       Random 2x128 frame + LoRA"
            echo "  config/config_c1b_frame_4x128.yaml       Random 4x128 frame + LoRA"
            echo "  config/config_c2_frame_higher_rank.yaml   Frame + LoRA r=16 (Phase 2)"
            echo "  config/config_c3_frame_partial_unfreeze.yaml  Frame + partial unfreeze (Phase 2)"
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
    [[ "${DRY_RUN}" == "true" ]] && EXTRA_ARGS+=(--dry-run)
    [[ "${FAST_DEV_RUN}" == "true" ]] && EXTRA_ARGS+=(--fast-dev-run)
    [[ "${GPU_CHECK}" -gt 0 ]] && EXTRA_ARGS+=(--gpu-check "${GPU_CHECK}")
    [[ -n "${NUM_WORKERS}" ]] && EXTRA_ARGS+=(--num-workers "${NUM_WORKERS}")
    [[ "${PROJECT_NAME}" != "ps-clef2026_birdclef-0" ]] && EXTRA_ARGS+=(--project-name "${PROJECT_NAME}")

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
    echo "Usage: $0 [CONFIG_PATH | --all] [--fast-dev-run] [--dry-run]"
    echo ""
    echo "Available configs:"
    echo "  config/config_a_baseline.yaml            Pretrained 16x16 (frozen) + LoRA"
    echo "  config/config_b_control.yaml             Random 16x16 (trainable) + LoRA"
    echo "  config/config_c1a_frame_2x128.yaml       Random 2x128 frame + LoRA"
    echo "  config/config_c1b_frame_4x128.yaml       Random 4x128 frame + LoRA"
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

# Extract run name from config filename (e.g., config_a_baseline)
RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Experiment 008: Frame Tokenization EAT"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
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
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},GPU_CHECK=${GPU_CHECK},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp008-${RUN_NAME} \
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
