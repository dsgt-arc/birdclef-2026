#!/bin/bash
#SBATCH --job-name=014-finetune
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t720 -qinferno
#SBATCH --output=logs/finetune-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export PYTHONUNBUFFERED=1

cd $repo_dir/user/anthony/experiments/014-efficientnet-soft-labels

module load cuda/12.1.1

echo "=== EfficientNet-B1 Soft Labels ==="
echo "Config: configs/finetune-soft.yaml"

uv run --project $repo_dir/user/anthony birdclef-efficientnet-soft \
    --config configs/finetune-soft.yaml "$@"
