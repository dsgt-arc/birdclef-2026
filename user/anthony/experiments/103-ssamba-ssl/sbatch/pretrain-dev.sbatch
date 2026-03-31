#!/bin/bash
#SBATCH --job-name=103-ssl-dev
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t30 -qinferno
#SBATCH --output=logs/pretrain-dev-%j.out

# Two-phase sanity check:
#   1. fast_dev_run: verify pipeline connects (1 train + 1 val batch)
#   2. overfit: train on 4 batches for 200 epochs — loss should drop to near zero

set -eu

CONFIG=${1:-configs/pretrain-103a.yaml}

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

# Persistent venv on scratch (first run builds, subsequent runs reuse)
source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== Phase 1: fast_dev_run (pipeline sanity check) ==="
birdclef-ssl-pretrain --config "$CONFIG" --fast-dev-run

echo ""
echo "=== Phase 2: overfit 4 batches for 200 epochs ==="
birdclef-ssl-pretrain --config "$CONFIG" --overfit-batches 4 \
    --ckpt-dir logs/overfit/checkpoints
