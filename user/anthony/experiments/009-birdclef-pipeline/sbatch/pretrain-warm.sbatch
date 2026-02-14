#!/bin/bash
#SBATCH --job-name=009-pretrain-warm
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1440 -qinferno
#SBATCH --output=logs/pretrain-warm-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

ckpt_dir=logs/pretrain-warm/checkpoints
pretrained=$scratch_dir/checkpoints/ssamba/ssamba_tiny_400.pth

echo "=== Warm-start pretrain: L1 + linear decoder + AudioSet backbone ==="
echo "Config: configs/pretrain-warm.yaml"
echo "Pre-trained backbone: ${pretrained}"
echo "Checkpoint dir: ${ckpt_dir}"

# --pretrained only used on first run; on resume, --ckpt-path last
# restores full state from Lightning checkpoint (ignores --pretrained)
$venv/bin/birdclef-pretrain \
    --config configs/pretrain-warm.yaml \
    --ckpt-dir ${ckpt_dir} \
    --ckpt-path last \
    --pretrained ${pretrained}
