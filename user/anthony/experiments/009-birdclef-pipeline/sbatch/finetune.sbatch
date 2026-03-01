#!/bin/bash
#SBATCH --job-name=009-finetune
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

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

pretrained_ckpt=$scratch_dir/checkpoints/ssamba/ssamba_tiny_400.pth

echo "=== Fine-tune: AudioSet backbone + SpecAugment ==="
echo "Config: configs/finetune.yaml"
echo "Pre-trained: $pretrained_ckpt"

uv run --project $repo_dir/user/anthony birdclef-finetune \
    --config configs/finetune.yaml \
    --pretrained-ckpt $pretrained_ckpt
