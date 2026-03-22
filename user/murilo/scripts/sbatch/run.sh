#!/bin/bash
# Launcher for soundscape inference jobs.
#
# Usage:
#   cd user/murilo/scripts
#   bash sbatch/run.sh --model all --split labeled
#   bash sbatch/run.sh --model protoclr-2026-soundscape --split all
#   bash sbatch/run.sh --model all --split labeled --dry-run

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$(dirname "$SCRIPTS_DIR")/logs"
JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

mkdir -p "$LOG_DIR"

# Parse args
DRY_RUN=false
MODEL=""
SPLIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 --model MODEL --split SPLIT [--dry-run]"
            echo ""
            echo "  --model   Model slug or 'all' (protoclr-2026-soundscape, convnext-2026-soundscape, eat-2026-soundscape)"
            echo "  --split   labeled, unlabeled, or all"
            echo "  --dry-run Preview command without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$SPLIT" ]]; then
    echo "Error: --model and --split are required"
    echo "Usage: $0 --model all --split labeled"
    exit 1
fi

echo "=============================================="
echo "Soundscape Inference"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Split: ${SPLIT}"
echo "Logs: ${LOG_DIR}/"
echo ""

CMD="sbatch \
    --output=${LOG_DIR}/slurm_infer_${SPLIT}_%j.log \
    --export=ALL,MODEL=${MODEL},SPLIT=${SPLIT} \
    ${JOB_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "View logs in: ${LOG_DIR}/"
fi
