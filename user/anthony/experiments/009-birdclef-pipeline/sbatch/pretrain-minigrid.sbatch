#!/bin/bash
#SBATCH --job-name=009-minigrid
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t120 -qinferno
#SBATCH --output=logs/pretrain-minigrid-%A-%a.out
#SBATCH --array=0-2

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

# Map array index to config
# A=baseline (variable spans, MSE, MLP), G=L1 loss, H=linear decoder
configs=(configs/minigrid-a.yaml configs/minigrid-g.yaml configs/minigrid-h.yaml)
labels=("A:baseline" "G:l1-loss" "H:linear-decoder")

config=${configs[$SLURM_ARRAY_TASK_ID]}
label=${labels[$SLURM_ARRAY_TASK_ID]}

echo "=== Mini-grid ${label} ==="
echo "Config: ${config}"

$venv/bin/birdclef-pretrain \
    --config ${config}
