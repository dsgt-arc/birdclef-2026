#!/bin/bash
#
# Submit Perch v2 TF inference SLURM job
#
# Usage:
#   cd user/murilo/inference
#   bash sbatch/run_perch.sh --source train_soundscapes
#   bash sbatch/run_perch.sh --source all --batch-size 16
#   bash sbatch/run_perch.sh --source train_soundscapes --max-files 100 --dry-run
#

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
JOB_SCRIPT="$SCRIPT_DIR/perch_inference.sh"

# Defaults
DRY_RUN=false
SOURCE=""
BATCH_SIZE="32"
MAX_FILES=""
OVERWRITE="false"
PROJECT_NAME="ps-clef2026_birdclef-0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --max-files)
            MAX_FILES="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE="true"; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --project-name)
            PROJECT_NAME="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$SOURCE" ]]; then
    echo "Error: --source is required (train_soundscapes | train_audio | all)"
    exit 1
fi

# Check if output already exists
OUTPUT_BASE="/storage/project/${PROJECT_NAME}/shared/inference/birdclef-2026/perch-v2-tf"
if [[ "$SOURCE" == "all" ]]; then
    CHECK_FILE="${OUTPUT_BASE}/train_soundscapes/embeddings.parquet"
else
    CHECK_FILE="${OUTPUT_BASE}/${SOURCE}/embeddings.parquet"
fi

if [[ -f "$CHECK_FILE" && "$OVERWRITE" != "true" ]]; then
    echo "Partial output exists: $CHECK_FILE — job will resume from last checkpoint."
fi

# Log directory
LOG_DIR="$INFERENCE_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Perch v2 TF Inference Submission"
echo "=============================================="
echo "Source: ${SOURCE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max files: ${MAX_FILES:-all}"
echo "Overwrite: ${OVERWRITE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Log dir: ${LOG_DIR}"
echo "=============================================="

CMD="sbatch \
    --output=${LOG_DIR}/slurm_perch_v2_tf_${SOURCE}_%j.log \
    --export=ALL,SOURCE=${SOURCE},BATCH_SIZE=${BATCH_SIZE},MAX_FILES=${MAX_FILES},OVERWRITE=${OVERWRITE},PROJECT_NAME=${PROJECT_NAME} \
    --job-name=perch-v2-tf-${SOURCE} \
    ${JOB_SCRIPT}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY RUN] Would execute:"
    echo "  $CMD"
else
    echo ""
    eval $CMD
fi
