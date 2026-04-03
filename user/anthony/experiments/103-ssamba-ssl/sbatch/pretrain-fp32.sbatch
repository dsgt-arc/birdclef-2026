#!/bin/bash
#SBATCH --job-name=103-ssl-pretrain
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1200 -qinferno
#SBATCH --output=logs/pretrain-fp32-%j.out

# RTX 6000 fp32 full training run.
# ~5-6h/epoch, 15 epochs total — will need multiple jobs with checkpoint resume.
# Usage:
#   sbatch sbatch/pretrain-fp32.sbatch
#   sbatch sbatch/pretrain-fp32.sbatch configs/pretrain-103a-fp32.yaml --ckpt-path last

set -eu

CONFIG=${1:-configs/pretrain-103a-fp32.yaml}
shift 2>/dev/null || true

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== SSL Pretraining (RTX 6000 fp32) ==="
echo "Config: $CONFIG"
echo "Extra args: $@"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

birdclef-ssl-pretrain --config "$CONFIG" "$@"
