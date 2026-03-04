#!/bin/bash
# Launcher script for inference jobs.
#
# Usage:
#   cd user/murilo/inference
#
#   # Submit inference for train_audio
#   bash sbatch/run.sh config/protoclr_lora_attn.yaml --source train_audio
#
#   # Submit inference for all sources
#   bash sbatch/run.sh config/protoclr_lora_attn.yaml --source all
#
#   # Debug with limited files
#   bash sbatch/run.sh config/protoclr_lora_attn.yaml --source train_audio --max-files 10
#
#   # Dry run (preview sbatch command)
#   bash sbatch/run.sh config/protoclr_lora_attn.yaml --source all --dry-run
#
# Flags:
#   CONFIG_PATH          Path to config YAML (required)
#   --source SOURCE      Data source: train_audio, train_soundscapes, all (required)
#   --max-files N        Limit number of files (for debugging)
#   --num-workers N      Override num_workers from config
#   --dry-run            Preview command without submitting job
#   --project-name NAME  PACE project allocation (default: ps-clef2026_birdclef-0)
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
CONFIG=""
SOURCE=""
MAX_FILES=""
NUM_WORKERS=""
PROJECT_NAME="ps-clef2026_birdclef-0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 CONFIG_PATH --source SOURCE [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  CONFIG_PATH       Path to config file (e.g., config/protoclr_lora_attn.yaml)"
            echo "  --source SOURCE   Data source: train_audio, train_soundscapes, all"
            echo ""
            echo "Options:"
            echo "  --max-files N     Limit number of files (for debugging)"
            echo "  --num-workers N   Override num_workers from config"
            echo "  --project-name    PACE project allocation name (default: ps-clef2026_birdclef-0)"
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

# Validate required args
if [[ -z "$CONFIG" ]]; then
    echo "Error: CONFIG_PATH is required"
    echo "Usage: $0 CONFIG_PATH --source SOURCE [OPTIONS]"
    exit 1
fi

if [[ -z "$SOURCE" ]]; then
    echo "Error: --source is required (train_audio, train_soundscapes, all)"
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

# Extract run name from config filename
RUN_NAME="$(basename "$CONFIG" .yaml)"

echo "=============================================="
echo "Inference: Embedding & Prediction Extraction"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Source: ${SOURCE}"
echo "Max files: ${MAX_FILES:-all}"
echo "Project: ${PROJECT_NAME}"
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
CPUS_OVERRIDE=""
if [[ -n "$NUM_WORKERS" ]]; then
    CPUS_OVERRIDE="--cpus-per-task=${NUM_WORKERS}"
fi

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_${SOURCE}_%j.log \
    --export=ALL,CONFIG=${CONFIG},PROJECT_NAME=${PROJECT_NAME},SOURCE=${SOURCE},MAX_FILES=${MAX_FILES},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=infer-${RUN_NAME} \
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
