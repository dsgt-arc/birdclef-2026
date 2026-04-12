#!/bin/bash
#SBATCH --job-name=exp107-sweep-lfam
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%j.out

# Pre-registered lambda_fam ablation on S22 (decision fold).
# Runs all 5 values sequentially in a single job, loading data once.
#
# lambda_fam values: {0, 1e-3, 1e-2, 1e-1, 1}
#
# Decision rule (from proposal): drop L_family in v2 if no lambda_fam > 0
# beats lambda_fam = 0 by >= 0.5 AUROC on amphibian subset.

set -euo pipefail

module load cuda/12.6.1

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"
mkdir -p logs

source sbatch/_setup-venv.sh

echo "=== lambda_fam sweep on S22 ==="
echo "=== Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="

python scripts/sweep_lambda_fam.py \
    --fold S22 \
    --lambda-site 0.1 \
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 256 \
    --checkpoint-root "$HOME/scratch/birdclef-2026/checkpoints/exp107-sweep" \
    --features-dir "$HOME/scratch/birdclef-2026/features/exp106"
