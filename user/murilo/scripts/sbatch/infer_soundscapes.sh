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
# Usage (from repo root or user/murilo/):
#   bash user/murilo/scripts/sbatch/infer_soundscapes.sh --model all --split labeled
#   bash user/murilo/scripts/sbatch/infer_soundscapes.sh --model protoclr-2026-soundscape --split all
#   bash user/murilo/scripts/sbatch/infer_soundscapes.sh --model all --split labeled --dry-run

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
USER_DIR="$(dirname "$SCRIPTS_DIR")"
INFER_SCRIPT="${SCRIPTS_DIR}/infer_soundscapes.py"
LOG_DIR="${USER_DIR}/logs"

# Detect launcher vs job mode: SCRIPT_ARGS is only set when we submit ourselves
if [[ -z "${SCRIPT_ARGS+x}" ]]; then
    # === Launcher mode: parse args and submit via sbatch ===
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

    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

    echo "=============================================="
    echo "Soundscape Inference"
    echo "=============================================="
    echo "Args: ${ARGS[*]}"
    echo "Logs: ${LOG_DIR}/"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] sbatch --output=${LOG_DIR}/slurm_infer_%j.log ${SCRIPT_PATH}"
        echo "  SCRIPT_ARGS=${ARGS[*]}"
    else
        echo "Submitting job..."
        export SCRIPT_ARGS="${ARGS[*]}"
        sbatch \
            --output="${LOG_DIR}/slurm_infer_%j.log" \
            --export=ALL \
            "${SCRIPT_PATH}"
    fi
    exit 0
fi

# === Job mode: running inside SLURM batch job ===
echo "=============================================="
echo "Soundscape Inference Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Args: ${SCRIPT_ARGS}"
echo "Scripts dir: ${SCRIPTS_DIR}"
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

VENV_BASE="${TMPDIR:-/tmp}"
JOB_VENV_DIR="$VENV_BASE/birdclef-${SLURM_JOB_ID}"
JOB_VENV="$JOB_VENV_DIR/.venv"
export UV_CACHE_DIR="$VENV_BASE/uv-cache"

echo "  TMPDIR: $VENV_BASE"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing dependencies from pyproject.toml..."
uv pip install -e "$SCRIPTS_DIR"

echo "  Installing CPU PyTorch..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "  Virtual environment ready!"
echo ""

# Verify
python -c "import onnxruntime; print(f'onnxruntime: {onnxruntime.__version__}')"
python -c "import librosa; print(f'librosa: {librosa.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
echo ""

# ============================================================================
# Run inference
# ============================================================================

python "$INFER_SCRIPT" ${SCRIPT_ARGS}

echo ""
echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
