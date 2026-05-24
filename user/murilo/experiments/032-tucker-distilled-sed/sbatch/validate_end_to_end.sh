#!/bin/bash
#SBATCH --job-name=exp032-e2e-val
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_e2e_val_%j.log
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Phase D § D.5 end-to-end validation — pre-flight check before LB submission.
# Runs the Kaggle kernel inference path on the training val fold and
# compares the resulting Tucker macro AUC against the training-side val/auroc.
#
# Usage:
#   cd user/murilo/experiments/032-tucker-distilled-sed
#
#   # Control checkpoint
#   CKPT=/storage/.../2026-exp032-distilled-sed-control/last.ckpt \
#   CONFIG=configs/distilled_sed_control.yaml \
#   TRAIN_AUROC=0.8928 \
#   sbatch sbatch/validate_end_to_end.sh
#
#   # KD checkpoint
#   CKPT=/storage/.../2026-exp032-distilled-sed-kd/last.ckpt \
#   CONFIG=configs/distilled_sed_kd.yaml \
#   sbatch sbatch/validate_end_to_end.sh
#
# Outputs:
#   logs/slurm_e2e_val_<jobid>.log  (includes the PASS/FAIL verdict)
#   outputs/e2e_val_<basename>.tsv  (per-class metrics, optional)

set -eu

CKPT="${CKPT:?CKPT must point to a .ckpt file}"
CONFIG="${CONFIG:?CONFIG must point to a .yaml config}"
TRAIN_AUROC="${TRAIN_AUROC:-}"
TOLERANCE="${TOLERANCE:-0.02}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Exp 032 Phase D § D.5 end-to-end validation"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Git commit:    $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Checkpoint:    ${CKPT}"
echo "Config:        ${CONFIG}"
echo "Train val/auroc (provided): ${TRAIN_AUROC:-<none>}"
echo "Tolerance:     ${TOLERANCE}"
echo ""
nvidia-smi
echo ""

# Setup venv (per CLAUDE.md: NEVER in home dir, use $TMPDIR)
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

uv pip install -e "$EXPERIMENT_DIR"
uv pip install "torch>=2.0.0,<2.11" "torchvision>=0.15,<0.26" "torchaudio>=2.0.0,<2.11" \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs outputs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

# Generate a stable output TSV name from the checkpoint path
CKPT_BASENAME="$(basename "$(dirname "$CKPT")")"
OUTPUT="outputs/e2e_val_${CKPT_BASENAME}.tsv"

EXTRA_ARGS=""
if [ -n "${TRAIN_AUROC}" ]; then
    EXTRA_ARGS="--training-side-auroc ${TRAIN_AUROC}"
fi

echo "=============================================="
echo "Running end-to-end validation..."
echo "=============================================="
srun python scripts/validate_end_to_end.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --output "${OUTPUT}" \
    --tolerance "${TOLERANCE}" \
    ${EXTRA_ARGS}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
