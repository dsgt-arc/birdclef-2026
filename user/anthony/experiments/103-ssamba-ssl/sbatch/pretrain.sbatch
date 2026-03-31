#!/bin/bash
#SBATCH --job-name=103-ssl-pretrain
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t720 -qinferno
#SBATCH --output=logs/pretrain-%j.out

# Usage:
#   sbatch sbatch/pretrain.sbatch
#   sbatch sbatch/pretrain.sbatch configs/pretrain-103a.yaml --ckpt-path last

set -eu

CONFIG=${1:-configs/pretrain-103a.yaml}
shift 2>/dev/null || true  # remaining args passed to CLI

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

# Persistent venv on scratch (first run builds, subsequent runs reuse)
source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== SSL Pretraining ==="
echo "Config: $CONFIG"
echo "Extra args: $@"

birdclef-ssl-pretrain --config "$CONFIG" "$@"
