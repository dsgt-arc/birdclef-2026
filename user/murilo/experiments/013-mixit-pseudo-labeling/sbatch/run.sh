#!/bin/bash
# Launcher script for Experiment 013 MixIT jobs.
#
# Usage:
#   cd user/murilo/experiments/013-mixit-pseudo-labeling
#
#   # Phase 1: Convert TF1 → ONNX
#   bash sbatch/run.sh convert
#
#   # Phase 2: Separate soundscapes
#   bash sbatch/run.sh separate                  # all unlabeled (default)
#   bash sbatch/run.sh separate --labeled         # labeled only (validation)
#   bash sbatch/run.sh separate --test            # 10 files (smoke test)
#
#   # Dry run
#   bash sbatch/run.sh convert --dry-run
#   bash sbatch/run.sh separate --dry-run

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

PHASE=""
DRY_RUN=false
MODE="unlabeled"

while [[ $# -gt 0 ]]; do
    case $1 in
        convert|separate)
            PHASE="$1"
            shift
            ;;
        --labeled)
            MODE="labeled"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 PHASE [OPTIONS]"
            echo ""
            echo "Phases:"
            echo "  convert             Phase 1: Convert Bird MixIT TF1 → ONNX"
            echo "  separate            Phase 2: Run MixIT separation on soundscapes"
            echo ""
            echo "Options (separate only):"
            echo "  --labeled           Process only labeled soundscapes (validation)"
            echo "  --test              Process 10 files only (smoke test)"
            echo ""
            echo "Options:"
            echo "  --dry-run           Preview command without submitting"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$PHASE" ]]; then
    echo "Error: PHASE is required (convert or separate)"
    echo "Usage: $0 PHASE [OPTIONS]"
    echo "  $0 convert              Phase 1: TF1 → ONNX conversion"
    echo "  $0 separate [--labeled|--test]  Phase 2: batch separation"
    exit 1
fi

# ============================================================================
# Determine paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MIXIT_DIR="/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit"

# ============================================================================
# Submit job
# ============================================================================

case "$PHASE" in
    convert)
        echo "=============================================="
        echo "Experiment 013: MixIT Phase 1 — TF1 → ONNX"
        echo "=============================================="
        echo "Experiment dir: ${EXPERIMENT_DIR}"
        echo "Checkpoint dir: ${MIXIT_DIR}"
        echo "Output: ${MIXIT_DIR}/bird_mixit_4sources.onnx"
        echo ""

        JOB_SCRIPT="${SCRIPT_DIR}/convert.sh"
        CMD="sbatch \
            --output=${LOG_DIR}/slurm_convert_%j.log \
            --job-name=exp013-convert \
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
        ;;

    separate)
        echo "=============================================="
        echo "Experiment 013: MixIT Phase 2 — Separation"
        echo "=============================================="
        echo "Experiment dir: ${EXPERIMENT_DIR}"
        echo "ONNX model: ${MIXIT_DIR}/bird_mixit_4sources.onnx"
        echo "Mode: ${MODE}"
        echo ""

        # Verify ONNX model exists
        if [[ ! -f "${MIXIT_DIR}/bird_mixit_4sources.onnx" ]]; then
            echo "WARNING: ONNX model not found. Run Phase 1 first:"
            echo "  bash sbatch/run.sh convert"
            echo ""
        fi

        JOB_SCRIPT="${SCRIPT_DIR}/separate.sh"
        CMD="sbatch \
            --output=${LOG_DIR}/slurm_separate_${MODE}_%j.log \
            --export=ALL,MODE=${MODE} \
            --job-name=exp013-sep-${MODE} \
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
        ;;
esac
