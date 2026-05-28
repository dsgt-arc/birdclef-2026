#!/bin/bash
#SBATCH --job-name=exp034-gate2
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 034 Phase D — Gate 2: frozen-backbone head-only sanity (NO LB).
#   Loads the Sydorskyi pretrained backbone (FROZEN), random-init SED head,
#   trains the head only on fold 0 for MAX_EPOCHS, reports OOF val/auroc on the
#   fold-0 holdout labeled-soundscape pool. Target >= 0.95 (Sydorskyi ref 0.984).
#
# Usage:
#   cd user/murilo/experiments/034-pretrained-backbone
#   sbatch sbatch/gate2.sbatch
#   # overrides:
#   FOLD=0 MAX_EPOCHS=5 BACKBONE=tf_efficientnetv2_s_in21k BATCH_SIZE=64 sbatch sbatch/gate2.sbatch
#   # NFNet-L0 variant:
#   BACKBONE=eca_nfnet_l0 sbatch sbatch/gate2.sbatch

set -eu

FOLD="${FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
BACKBONE="${BACKBONE:-tf_efficientnetv2_s_in21k}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_birdclef-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

echo "=============================================="
echo "Exp 034 Phase D — Gate 2 (frozen-backbone head sanity)"
echo "=============================================="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Backbone: ${BACKBONE}   Fold: ${FOLD}   Epochs: ${MAX_EPOCHS}   BS: ${BATCH_SIZE}"
echo "Git:      $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown))"
echo ""
nvidia-smi
echo ""

# Per-job venv on fast local NVMe ($TMPDIR) — per CLAUDE.md venv policy.
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
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs outputs
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

RUN_NAME="gate2-${BACKBONE}"
echo "Starting Gate 2 head-only training (frozen backbone, pretrained loaded)..."
srun python train.py \
    --backbone "${BACKBONE}" \
    --fold "${FOLD}" \
    --freeze-backbone \
    --load-pretrained \
    --max-epochs "${MAX_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 8 \
    --accelerator gpu \
    --run-name "${RUN_NAME}"

echo ""
echo "=============================================="
echo "GATE 2 RESULT (target OOF val/auroc >= 0.95):"
cat "outputs/${RUN_NAME}-fold${FOLD}.json" 2>/dev/null || echo "  (metrics JSON not found — check the log above)"
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
