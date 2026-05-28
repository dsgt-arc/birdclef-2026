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
#SBATCH --output=logs/slurm-gate2-%j.out
#
# Exp 034 Phase D — Gate 2: frozen-backbone head-only sanity (NO LB).
#   Loads Sydorskyi pretrained backbone (FROZEN) + random-init head, trains the
#   head only on fold 0 for MAX_EPOCHS, reports OOF val/auroc (target >= 0.95).
#
# RUN WITH (self-submits to SLURM):
#   cd user/murilo/experiments/034-pretrained-backbone
#   bash sbatch/gate2.sh
#   BACKBONE=eca_nfnet_l0 bash sbatch/gate2.sh      # NFNet-L0 variant
#   FOLD=0 MAX_EPOCHS=5 bash sbatch/gate2.sh         # overrides
#   (also works as: sbatch sbatch/gate2.sh)

set -eu

# --- self-submit: when invoked via `bash` (no SLURM_JOB_ID yet), resubmit via sbatch.
#     cd into the experiment dir first so $SLURM_SUBMIT_DIR points there. ---
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../sbatch
    EXP="$(dirname "$HERE")"
    mkdir -p "$EXP/logs"
    cd "$EXP"
    echo "Submitting Gate 2 to SLURM: sbatch $HERE/gate2.sh"
    exec sbatch --export=ALL "$HERE/gate2.sh"
fi

# =========================== running under SLURM ===========================
# NOTE: sbatch spools a COPY of this script to /var/lib/slurm/..., so
# ${BASH_SOURCE[0]} is NOT the repo path here. Use $SLURM_SUBMIT_DIR (the dir
# that `bash sbatch/gate2.sh` cd'd into = the experiment dir).
EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

FOLD="${FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
BACKBONE="${BACKBONE:-tf_efficientnetv2_s_in21k}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "=============================================="
echo "Exp 034 Phase D — Gate 2 (frozen-backbone head sanity)"
echo "=============================================="
echo "Job ID:    ${SLURM_JOB_ID}   Node: $(hostname)   Date: $(date)"
echo "ExpDir:    ${EXPERIMENT_DIR}"
echo "Backbone:  ${BACKBONE}   Fold: ${FOLD}   Epochs: ${MAX_EPOCHS}   BS: ${BATCH_SIZE}"
echo "Git:       $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown))"
echo ""
nvidia-smi || true
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
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs outputs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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
cat "outputs/${RUN_NAME}-fold${FOLD}.json" 2>/dev/null || echo "  (metrics JSON not found — check log above)"
echo "=============================================="
echo "Job completed at $(date)"
