#!/bin/bash
#SBATCH --job-name=103-ssl-pretrain
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu-a100
#SBATCH -t1200 -qinferno
#SBATCH --output=logs/pretrain-a100-%j.out

# A100 bf16 full training run.
# Usage:
#   sbatch sbatch/pretrain-a100.sbatch
#   sbatch sbatch/pretrain-a100.sbatch configs/pretrain-103a-a100.yaml --ckpt-path last

set -eu

CONFIG=${1:-configs/pretrain-103a-a100.yaml}
shift 2>/dev/null || true

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== SSL Pretraining (A100 bf16) ==="
echo "Config: $CONFIG"
echo "Extra args: $@"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

birdclef-ssl-pretrain --config "$CONFIG" "$@"
