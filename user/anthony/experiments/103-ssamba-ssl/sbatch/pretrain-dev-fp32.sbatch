#!/bin/bash
#SBATCH --job-name=103-ssl-fp32-dev
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t60 -qinferno
#SBATCH --output=logs/pretrain-dev-fp32-%j.out

# RTX 6000 fp32 overfit test:
#   1. fast_dev_run: verify fp32 pipeline works
#   2. overfit: train on 4 batches for 200 epochs — loss should drop, no NaN

set -eu

CONFIG=${1:-configs/pretrain-103a-fp32.yaml}

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== RTX 6000 fp32 overfit test ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

echo "=== Phase 1: fast_dev_run (pipeline sanity check) ==="
birdclef-ssl-pretrain --config "$CONFIG" --fast-dev-run

echo ""
echo "=== Phase 2: overfit 4 batches for 200 epochs ==="
birdclef-ssl-pretrain --config "$CONFIG" --overfit-batches 4 \
    --ckpt-dir logs/overfit-fp32/checkpoints
