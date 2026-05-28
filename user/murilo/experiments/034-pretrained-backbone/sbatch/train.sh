#!/bin/bash
#SBATCH --job-name=exp034-train
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#SBATCH --output=logs/slurm-train-%j.out
#
# Exp 034 Phase E/F/G — XCPretrain fine-tune of the Sydorskyi backbone on 2026.
#   MODE=frozen  → freeze backbone, head-only (Phase E first LB, Phase F folds).
#                  Default 25 epochs (our proven exp 032 cadence; head converges
#                  fast on fixed features + cosine anneals over the actual run).
#   MODE=fullft  → unfreeze backbone (Phase G). Default 50 epochs (Sydorskyi),
#                  grad-clip 1.0 for stability.
#
# RUN WITH (self-submits to SLURM):
#   cd user/murilo/experiments/034-pretrained-backbone
#   # Phase E — frozen-backbone, fold 0, 25 ep:
#   bash sbatch/train.sh
#   # 50-ep ablation (cheap; let LB arbitrate 25 vs 50):
#   MAX_EPOCHS=50 RUN_SUFFIX=-50ep bash sbatch/train.sh
#   # Phase F — all 5 folds (submit one per fold):
#   for f in 0 1 2 3 4; do FOLD=$f bash sbatch/train.sh; done
#   # Phase G — full fine-tune:
#   MODE=fullft bash sbatch/train.sh
#   # NFNet-L0:
#   BACKBONE=eca_nfnet_l0 bash sbatch/train.sh

set -eu

# Self-submit unless WE already did (custom sentinel — NOT $SLURM_JOB_ID, which
# is also set inside interactive salloc/srun and would skip submission).
if [[ -z "${XCPRETRAIN_SUBMITTED:-}" ]]; then
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXP="$(dirname "$HERE")"
    mkdir -p "$EXP/logs"
    cd "$EXP"
    echo "Submitting train to SLURM: sbatch $HERE/train.sh"
    exec sbatch --export=ALL,XCPRETRAIN_SUBMITTED=1 "$HERE/train.sh"
fi

# Hard guard: the heavy body must only run inside a SLURM batch job.
: "${SLURM_JOB_ID:?train.sh body reached outside SLURM — launch as 'bash sbatch/train.sh'}"

# =========================== running under SLURM ===========================
EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

MODE="${MODE:-frozen}"          # frozen | fullft
FOLD="${FOLD:-0}"
BACKBONE="${BACKBONE:-tf_efficientnetv2_s_in21k}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

if [[ "$MODE" == "frozen" ]]; then
    MAX_EPOCHS="${MAX_EPOCHS:-25}"   # proven cadence; head-only converges fast
    FREEZE_FLAG="--freeze-backbone"
    GRAD_CLIP="${GRAD_CLIP:-0.0}"     # Sydorskyi: none
elif [[ "$MODE" == "fullft" ]]; then
    MAX_EPOCHS="${MAX_EPOCHS:-50}"   # Sydorskyi full-FT
    FREEZE_FLAG=""
    GRAD_CLIP="${GRAD_CLIP:-1.0}"     # stability for whole-network FT
else
    echo "ERROR: MODE must be 'frozen' or 'fullft', got '$MODE'"; exit 1
fi

RUN_NAME="xcpretrain-${BACKBONE}-${MODE}${RUN_SUFFIX}"

echo "=============================================="
echo "Exp 034 — XCPretrain fine-tune (${MODE})"
echo "=============================================="
echo "Job ID:    ${SLURM_JOB_ID}   Node: $(hostname)   Date: $(date)"
echo "ExpDir:    ${EXPERIMENT_DIR}"
echo "Run:       ${RUN_NAME}   Backbone: ${BACKBONE}   Fold: ${FOLD}"
echo "Epochs:    ${MAX_EPOCHS}   BS: ${BATCH_SIZE}   grad_clip: ${GRAD_CLIP}   freeze: '${FREEZE_FLAG}'"
echo "Git:       $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown))"
echo ""
nvidia-smi || true
echo ""

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

# Load W&B / other creds if present (optional).
if [[ -f "$USER_DIR/.env" ]]; then set -a; source "$USER_DIR/.env"; set +a; fi

echo "Starting ${MODE} fine-tune..."
srun python train.py \
    --backbone "${BACKBONE}" \
    --fold "${FOLD}" \
    ${FREEZE_FLAG} \
    --load-pretrained \
    --max-epochs "${MAX_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 8 \
    --grad-clip "${GRAD_CLIP}" \
    --save-ckpt \
    --accelerator gpu \
    --run-name "${RUN_NAME}" \
    ${WANDB:+--wandb}

echo ""
echo "=============================================="
echo "RESULT (val/auroc):"
cat "outputs/${RUN_NAME}-fold${FOLD}.json" 2>/dev/null || echo "  (metrics JSON not found — check log above)"
echo "Checkpoints: ${CKPT_ROOT:-/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/034-pretrained-backbone/checkpoints}/${RUN_NAME}-fold${FOLD}/"
echo "=============================================="
echo "Job completed at $(date)"
