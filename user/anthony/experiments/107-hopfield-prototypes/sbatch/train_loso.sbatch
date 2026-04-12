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
LAMBDA_SITE=${LAMBDA_SITE:-0.1}
LAMBDA_FAM=${LAMBDA_FAM:-0.01}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-3}
BATCH_SIZE=${BATCH_SIZE:-256}

echo "=== Training fold: $FOLD ==="
echo "=== Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="
echo "=== lambda_site=$LAMBDA_SITE lambda_fam=$LAMBDA_FAM epochs=$EPOCHS ==="

python scripts/train.py \
    --fold "$FOLD" \
    --lambda-site "$LAMBDA_SITE" \
    --lambda-fam "$LAMBDA_FAM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-dir "$HOME/scratch/birdclef-2026/checkpoints/exp107" \
    --features-dir "$HOME/scratch/birdclef-2026/features/exp106"
