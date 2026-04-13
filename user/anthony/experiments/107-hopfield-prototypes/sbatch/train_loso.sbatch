#!/bin/bash
#SBATCH --job-name=exp107-loso
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%j.out

# LOSO training: single job that loads data once and iterates over folds
# sequentially.  Data loading is ~9 min and dominates runtime; running
# folds sequentially in-memory is ~5x faster than an array job.
#
# Usage:
#   sbatch sbatch/train_loso.sbatch                 # smoke: S22 only
#   FOLD=all sbatch sbatch/train_loso.sbatch        # full 9-fold LOSO

set -euo pipefail

module load cuda/12.6.1

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

FOLD=${FOLD:-S22}
LAMBDA_SITE=${LAMBDA_SITE:-0.01}
LAMBDA_XATTN=${LAMBDA_XATTN:-0.01}
LAMBDA_FAM=${LAMBDA_FAM:-0.01}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-4}
BATCH_SIZE=${BATCH_SIZE:-256}
CKPT_SUBDIR=${CKPT_SUBDIR:-exp107}

echo "=== Training fold: $FOLD ==="
echo "=== Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="
echo "=== lambda_site=$LAMBDA_SITE lambda_xattn=$LAMBDA_XATTN lambda_fam=$LAMBDA_FAM epochs=$EPOCHS lr=$LR ==="
echo "=== ckpt_subdir=$CKPT_SUBDIR ==="

# Unbuffered stdout so progress prints stream to the log file
export PYTHONUNBUFFERED=1

# Strict CUDA precondition: fail fast if the driver can't init, so we
# don't silently burn the walltime on a CPU fallback. Job 6484295 hit
# this — venv sanity check said True, then train.py's own init failed.
python -u -c "
import sys, torch
if not torch.cuda.is_available():
    print('FATAL: CUDA not available at training-entry.', file=sys.stderr)
    print(f'torch={torch.__version__} cuda={torch.version.cuda}', file=sys.stderr)
    sys.exit(2)
print(f'CUDA ok: device={torch.cuda.get_device_name(0)}')
"

python -u scripts/train.py \
    --fold "$FOLD" \
    --lambda-site "$LAMBDA_SITE" \
    --lambda-xattn "$LAMBDA_XATTN" \
    --lambda-fam "$LAMBDA_FAM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-dir "$HOME/scratch/birdclef-2026/checkpoints/$CKPT_SUBDIR" \
    --features-dir "$HOME/scratch/birdclef-2026/features/exp106"
