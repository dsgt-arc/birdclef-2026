#!/bin/bash
#SBATCH --job-name=103c-linprobe
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t120 -qinferno
#SBATCH --output=logs/linprobe-103c-%j.out

# Linear probe evaluation of SSL pretrained SSAMBA-tiny.
# Should converge in <1h (only 45K trainable params).

set -eu

CONFIG=${1:-configs/linprobe-103c.yaml}

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== 103c Linear Probe ==="
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

birdclef-ssl-linprobe --config "$CONFIG"
