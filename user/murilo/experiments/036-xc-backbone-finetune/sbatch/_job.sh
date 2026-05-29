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
# Exp 036 shared SLURM job (matches exp 032/033/034 _job.sh pattern).
# job-name / partition / output are set by the launcher's sbatch flags.
# DO NOT run directly — use the launcher:  bash sbatch/train.sh  (CONFIG=configs/<cell>.yaml)
#
# Params via env (set by the launcher's `sbatch --export`):
#   CONFIG (required, path to configs/*.yaml), FOLD, FAST_DEV_RUN (true/false),
#   NUM_WORKERS (optional override).

set -eu
: "${SLURM_JOB_ID:?_job.sh must run via sbatch — use 'bash sbatch/gate2.sh' or 'bash sbatch/train.sh'}"
: "${CONFIG:?CONFIG env var required (path to a configs/*.yaml)}"

# sbatch spools a COPY of this script to /var/lib/slurm/..., so $BASH_SOURCE is
# NOT the repo path. $SLURM_SUBMIT_DIR is the dir the launcher submitted from.
EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
REPO_DIR="$(dirname "$(dirname "$USER_DIR")")"

FOLD="${FOLD:-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
NUM_WORKERS="${NUM_WORKERS:-}"

echo "=============================================="
echo "Exp 036 _job.sh"
echo "=============================================="
echo "Job ID:   ${SLURM_JOB_ID}   Node: $(hostname)   Date: $(date)"
echo "ExpDir:   ${EXPERIMENT_DIR}"
echo "Config:   ${CONFIG}   Fold: ${FOLD}   FastDevRun: ${FAST_DEV_RUN}"
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

ARGS=(--config "$CONFIG" --fold "$FOLD")
[ "$FAST_DEV_RUN" = "true" ] && ARGS+=(--fast-dev-run)
[ -n "$NUM_WORKERS" ] && ARGS+=(--num-workers "$NUM_WORKERS")

echo "Running: python train.py ${ARGS[*]}"
echo ""
srun python train.py "${ARGS[@]}"

# Structured metrics → artifacts/ (skip for fast-dev-run; matches exp 032).
if [ "$FAST_DEV_RUN" != "true" ]; then
    echo ""
    echo "Saving structured metrics to artifacts/..."
    python save_metrics.py --config "$CONFIG" --fold "$FOLD"
fi

echo ""
echo "=============================================="
echo "Job completed at $(date)"
