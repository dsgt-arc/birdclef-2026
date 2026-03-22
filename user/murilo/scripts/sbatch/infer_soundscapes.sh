#!/bin/bash
#SBATCH --job-name=infer-soundscapes
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Run ONNX inference on soundscapes.
#
# Usage:
#   # Labeled soundscapes, all 3 models
#   bash user/murilo/scripts/sbatch/infer_soundscapes.sh --model all --split labeled
#
#   # Single model, all soundscapes
#   bash user/murilo/scripts/sbatch/infer_soundscapes.sh --model protoclr-2026-soundscape --split all
#
#   # Dry run
#   bash user/murilo/scripts/sbatch/infer_soundscapes.sh --model all --split labeled --dry-run

set -eu

# When run via sbatch, args come from SCRIPT_ARGS env var
# When run directly (as launcher), we parse args and submit via sbatch
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # === Launcher mode: parse args and submit ===
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs"
    mkdir -p "$LOG_DIR"

    DRY_RUN=false
    ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == "--dry-run" ]]; then
            DRY_RUN=true
        else
            ARGS+=("$arg")
        fi
    done

    CMD="sbatch \
        --output=${LOG_DIR}/slurm_infer_%j.log \
        --export=ALL,SCRIPT_ARGS=\"${ARGS[*]}\" \
        ${SCRIPT_PATH}"

    echo "=============================================="
    echo "Soundscape Inference"
    echo "=============================================="
    echo "Args: ${ARGS[*]}"
    echo "Logs: ${LOG_DIR}/"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] ${CMD}"
    else
        echo "Submitting job..."
        ${CMD}
    fi
    exit 0
fi

# === Job mode: running inside SLURM ===
echo "=============================================="
echo "Soundscape Inference Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Args: ${SCRIPT_ARGS:-}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="$(dirname "$SCRIPT_DIR")/scripts/infer_soundscapes.py"

# Setup venv
JOB_VENV_DIR="$TMPDIR/birdclef-${SLURM_JOB_ID}"
JOB_VENV="$JOB_VENV_DIR/.venv"
export UV_CACHE_DIR="$TMPDIR/uv-cache"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "Installing dependencies..."
uv pip install onnxruntime soundfile librosa numpy tqdm torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo ""

# Run inference
python "$INFER_SCRIPT" ${SCRIPT_ARGS:-}

echo ""
echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
