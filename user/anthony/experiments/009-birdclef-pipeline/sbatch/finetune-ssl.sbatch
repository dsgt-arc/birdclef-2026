#!/bin/bash
#SBATCH --job-name=009-finetune-ssl
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1440 -qinferno
#SBATCH --output=logs/finetune-ssl-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

pretrained_ckpt=logs/pretrain-full/checkpoints/last.ckpt
last_ckpt=logs/finetune/checkpoints/last.ckpt

echo "=== Fine-tune: SSL pretrain-full backbone ==="
echo "Config: configs/finetune.yaml"

if [ -f "$last_ckpt" ]; then
    echo "Resuming from $last_ckpt"
    $venv/bin/birdclef-finetune \
        --config configs/finetune.yaml \
        --ckpt-path "$last_ckpt"
else
    echo "Starting fresh with pretrained backbone: $pretrained_ckpt"
    $venv/bin/birdclef-finetune \
        --config configs/finetune.yaml \
        --pretrained-ckpt "$pretrained_ckpt"
fi
