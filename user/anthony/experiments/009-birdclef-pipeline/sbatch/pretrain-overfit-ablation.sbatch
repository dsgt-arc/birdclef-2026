#!/bin/bash
#SBATCH --job-name=009-ovf-abl
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t60 -qinferno
#SBATCH --output=logs/pretrain-overfit-ablation-%A-%a.out
#SBATCH --array=0-2

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
venv=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

# Map array index to config
configs=(configs/overfit-a.yaml configs/overfit-b.yaml configs/overfit-d.yaml)
labels=("A:baseline" "B:no-norm" "D:low-mask")

config=${configs[$SLURM_ARRAY_TASK_ID]}
label=${labels[$SLURM_ARRAY_TASK_ID]}

echo "=== Condition ${label} ==="
echo "Config: ${config}"

$venv/bin/birdclef-pretrain \
    --config ${config} \
    --overfit-batches 4
