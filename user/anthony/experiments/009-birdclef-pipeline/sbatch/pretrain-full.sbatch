#!/bin/bash
#SBATCH --job-name=009-pretrain-full
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1440 -qinferno
#SBATCH --output=logs/pretrain-full-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

ckpt_dir=logs/pretrain-full/checkpoints

echo "=== Full pretrain: L1 + linear decoder ==="
echo "Config: configs/pretrain-full.yaml"
echo "Checkpoint dir: ${ckpt_dir}"

# Auto-resume from last checkpoint if it exists
$venv/bin/birdclef-pretrain \
    --config configs/pretrain-full.yaml \
    --ckpt-dir ${ckpt_dir} \
    --ckpt-path last
