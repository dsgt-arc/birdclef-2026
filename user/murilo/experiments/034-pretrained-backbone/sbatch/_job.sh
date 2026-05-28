#!/bin/bash
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 034 shared SLURM job (matches exp 032/033 _job.sh pattern).
# job-name / partition / output are set by the launcher's sbatch flags.
# DO NOT run directly — use a launcher:  bash sbatch/gate2.sh  |  bash sbatch/train.sh
#
# Params arrive as env vars (set by the launcher via `sbatch --export`):
#   BACKBONE FOLD MAX_EPOCHS BATCH_SIZE GRAD_CLIP RUN_NAME
#   FREEZE LOAD_PRETRAINED SAVE_CKPT WANDB NO_SPEC_AUGMENT   (boolean 0/1)

set -eu

# Hard guard: this body must only run inside a SLURM batch job.
: "${SLURM_JOB_ID:?_job.sh must run via sbatch — use 'bash sbatch/gate2.sh' or 'bash sbatch/train.sh'}"

# sbatch spools a COPY of this script to /var/lib/slurm/..., so $BASH_SOURCE is
# NOT the repo path. $SLURM_SUBMIT_DIR is the dir the launcher submitted from.
EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

BACKBONE="${BACKBONE:-tf_efficientnetv2_s_in21k}"
FOLD="${FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
RUN_NAME="${RUN_NAME:-xcpretrain}"

echo "=============================================="
echo "Exp 034 _job.sh"
echo "=============================================="
echo "Job ID:   ${SLURM_JOB_ID}   Node: $(hostname)   Date: $(date)"
echo "ExpDir:   ${EXPERIMENT_DIR}"
echo "Run:      ${RUN_NAME}   Backbone: ${BACKBONE}   Fold: ${FOLD}   Epochs: ${MAX_EPOCHS}"
echo "Flags:    FREEZE=${FREEZE:-0} LOAD_PRETRAINED=${LOAD_PRETRAINED:-0} SAVE_CKPT=${SAVE_CKPT:-0} grad_clip=${GRAD_CLIP}"
echo "Git:      $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown))"
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
[ -f "$USER_DIR/.env" ] && { set -a; source "$USER_DIR/.env"; set +a; }

# Build train.py args from boolean env flags (avoids spaces in --export values).
ARGS=(--backbone "$BACKBONE" --fold "$FOLD" --max-epochs "$MAX_EPOCHS"
      --batch-size "$BATCH_SIZE" --num-workers 8 --grad-clip "$GRAD_CLIP"
      --accelerator gpu --run-name "$RUN_NAME")
[ "${FREEZE:-0}" = "1" ]          && ARGS+=(--freeze-backbone)
[ "${LOAD_PRETRAINED:-0}" = "1" ] && ARGS+=(--load-pretrained)
[ "${SAVE_CKPT:-0}" = "1" ]       && ARGS+=(--save-ckpt)
[ "${WANDB:-0}" = "1" ]           && ARGS+=(--wandb)
[ "${NO_SPEC_AUGMENT:-0}" = "1" ] && ARGS+=(--no-spec-augment)

echo "Running: python train.py ${ARGS[*]}"
echo ""
srun python train.py "${ARGS[@]}"

echo ""
echo "=============================================="
echo "RESULT (val/auroc):"
cat "outputs/${RUN_NAME}-fold${FOLD}.json" 2>/dev/null || echo "  (metrics JSON not found — check log above)"
echo "=============================================="
echo "Job completed at $(date)"
