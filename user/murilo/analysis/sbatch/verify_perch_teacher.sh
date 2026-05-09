#!/bin/bash
#SBATCH --job-name=verify-perch-teacher
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -qinferno
#SBATCH --output=/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/analysis/logs/slurm_verify_perch_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Phase 0.1 verification — confirm CUDAExecutionProvider works for the
# vendored Perch v2 ONNX teacher on a real GPU node.
#
# Usage:
#   bash user/murilo/analysis/sbatch/verify_perch_teacher.sh
#
# When invoked via `bash`, the script self-submits via `sbatch` (the guard
# below). When SLURM runs the staged copy, the SBATCH directives kick in
# and we land in the actual job body.
#
# Logs written to: user/murilo/analysis/logs/slurm_verify_perch_<JOBID>.log

set -eu

# ---------------------------------------------------------------------------
# Self-submit guard: if we're not running under THIS specific sbatch job
# (recognized by --job-name=verify-perch-teacher), resubmit via sbatch.
#
# We check SLURM_JOB_NAME rather than SLURM_JOB_ID because the user may be
# running `bash` from inside an unrelated SLURM allocation (e.g. an
# interactive CPU salloc), in which case SLURM_JOB_ID is set but we're NOT
# inside the GPU job we want.
# ---------------------------------------------------------------------------
EXPECTED_JOB_NAME="verify-perch-teacher"
if [ "${SLURM_JOB_NAME:-}" != "${EXPECTED_JOB_NAME}" ]; then
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    echo "Not inside the '${EXPECTED_JOB_NAME}' sbatch job (current SLURM_JOB_NAME='${SLURM_JOB_NAME:-<unset>}')."
    echo "Submitting via sbatch: $SCRIPT_PATH"
    exec sbatch "$SCRIPT_PATH" "$@"
fi

PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"

# Hardcoded absolute paths — sbatch resolves $0 / SLURM_SUBMIT_DIR
# unreliably (SLURM_SUBMIT_DIR can be /var/lib/slurm/slurmd and $0 can be
# the staged copy of this script in slurmd's runtime dir, not the actual
# project paths).
REPO_DIR="/storage/project/${PROJECT_NAME}/mgustineli3/birdclef-2026"
USER_DIR="${REPO_DIR}/user/murilo"
ANALYSIS_DIR="${USER_DIR}/analysis"

echo "=============================================="
echo "Job: Verify Perch v2 ONNX teacher (Phase 0.1)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Analysis dir: $ANALYSIS_DIR"
echo "Git commit: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo ""
nvidia-smi
echo ""

# Setup venv (per-job, on local NVMe per CLAUDE.md)
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing local birdclef package..."
uv pip install -e "$USER_DIR"

echo "  Installing PyTorch with CUDA..."
uv pip install "torch>=2.0.0,<2.11" "torchvision>=0.15,<0.26" "torchaudio>=2.0.0,<2.11" --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: install onnxruntime-gpu (NOT plain onnxruntime). Plain onnxruntime
# is CPU-only — CUDAExecutionProvider would fail to load. If the local package
# already pulled in onnxruntime, the -gpu install will replace it.
echo "  Installing onnxruntime-gpu (replaces any CPU-only onnxruntime)..."
uv pip uninstall -y onnxruntime 2>/dev/null || true
uv pip install "onnxruntime-gpu>=1.18,<2.0"

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import onnxruntime as ort; print(f'onnxruntime: {ort.__version__}, providers: {ort.get_available_providers()}')"
echo ""

cd "$ANALYSIS_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the verification — same script that ran on CPU node, now with GPU available
echo "=============================================="
echo "Running verify_perch_teacher.py on GPU..."
echo "=============================================="
echo ""

srun python verify_perch_teacher.py

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job complete."
echo ""
echo "Schema doc: ${USER_DIR}/experiments/031-perch-kd-distillation/phase0_perch_teacher_schema.md"
echo "=============================================="
