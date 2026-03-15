#!/bin/bash
#SBATCH --job-name=exp010-validate
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 010: Local validation of ONNX exports (CPU only)
# Measures per-file inference time and validates submission format.
#
# Usage:
#   cd user/murilo/experiments/010-submission-pipeline
#   sbatch sbatch/validate.sbatch                 # validate all models
#   MODEL=eat sbatch sbatch/validate.sbatch       # validate specific model

set -eu

MODEL="${MODEL:-all}"

# ============================================================================
# Paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

echo "=============================================="
echo "Job: Experiment 010 - Local Validation (CPU)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Model: ${MODEL}"
echo ""

# ============================================================================
# Setup venv in TMPDIR
# ============================================================================

export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "Setting up venv at $JOB_VENV..."
mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

uv pip install -e "$EXPERIMENT_DIR"
# CPU-only torch for validation
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "Venv ready!"
echo ""

cd "$EXPERIMENT_DIR"

# ============================================================================
# Run validation
# ============================================================================

validate_model() {
    local model_name="$1"
    echo "=== Validating $model_name ==="
    python scripts/local_validate.py \
        --model "$model_name" \
        --max-files 10
    echo ""
}

case "$MODEL" in
    eat-2025|convnext-2025|protoclr-2025|eat-2026|convnext-2026|protoclr-2026)
        validate_model "$MODEL"
        ;;
    all-2025)
        for m in eat-2025 convnext-2025 protoclr-2025; do
            if [ -f "artifacts/weights/$m/model.onnx" ]; then
                validate_model "$m"
            else
                echo "Skipping $m (no ONNX model found)"
            fi
        done
        ;;
    all-2026)
        for m in eat-2026 convnext-2026 protoclr-2026; do
            if [ -f "artifacts/weights/$m/model.onnx" ]; then
                validate_model "$m"
            else
                echo "Skipping $m (no ONNX model found)"
            fi
        done
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

echo "=============================================="
echo "Validation complete! $(date)"
echo "=============================================="
