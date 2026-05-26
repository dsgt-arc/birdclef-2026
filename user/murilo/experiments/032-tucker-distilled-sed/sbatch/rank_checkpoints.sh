#!/bin/bash
#SBATCH --job-name=exp032-rank
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_rank_%j.log
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Post-hoc rank all 25 epoch checkpoints by Tucker's correct macro AUROC
# (skip-empty-classes) for both control + KD runs.
#
# Background: the in-training ModelCheckpoint monitor used torchmetrics'
# MultilabelAUROC(average='macro') which dragged val/cmap + val/auroc to
# ~0.11 because 205/234 classes have zero positives in the small SS val
# pool. All 25 epochs are saved (save_top_k=-1); this script re-ranks
# them by Tucker's compute_macro_auc (skips 0-positive classes) and
# recommends the best-by-Tucker epoch per run for LB submission.
#
# Usage:
#   cd user/murilo/experiments/032-tucker-distilled-sed
#   sbatch sbatch/rank_checkpoints.sh
#
# Outputs:
#   outputs/checkpoint_ranking_control.tsv
#   outputs/checkpoint_ranking_kd.tsv
#   logs/slurm_rank_<jobid>.log
#
# Override checkpoint dirs via env vars if needed:
#   CONTROL_CKPT_DIR=/path/to/control sbatch sbatch/rank_checkpoints.sh

set -eu

CONTROL_CKPT_DIR="${CONTROL_CKPT_DIR:-/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-control}"
KD_CKPT_DIR="${KD_CKPT_DIR:-/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd}"
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
echo "Exp 032: Post-hoc checkpoint ranking"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Git commit:    $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Git branch:    $(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo 'unknown')"
echo "Control dir:   ${CONTROL_CKPT_DIR}"
echo "KD dir:        ${KD_CKPT_DIR}"
echo ""
nvidia-smi
echo ""

# Setup venv (per CLAUDE.md: NEVER in home dir, use $TMPDIR)
echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing PyTorch with CUDA..."
uv pip install "torch>=2.0.0,<2.11" "torchvision>=0.15,<0.26" "torchaudio>=2.0.0,<2.11" --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""

cd "$EXPERIMENT_DIR"
mkdir -p logs outputs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/huggingface"
mkdir -p "$HF_HOME"

if [ -f "$USER_DIR/.env" ]; then
    set -a
    source "$USER_DIR/.env"
    set +a
fi

echo "=============================================="
echo "Ranking CONTROL checkpoints..."
echo "=============================================="
srun python scripts/score_all_checkpoints.py \
    --config configs/distilled_sed_control.yaml \
    --ckpt-dir "${CONTROL_CKPT_DIR}" \
    --output outputs/checkpoint_ranking_control.tsv \
    --num-workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "=============================================="
echo "Ranking KD checkpoints..."
echo "=============================================="
srun python scripts/score_all_checkpoints.py \
    --config configs/distilled_sed_kd.yaml \
    --ckpt-dir "${KD_CKPT_DIR}" \
    --output outputs/checkpoint_ranking_kd.tsv \
    --num-workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
