#!/bin/bash
#SBATCH --job-name=015-finetune
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

cd $repo_dir/user/anthony/experiments/015-efficientnet-sed

module load cuda/12.1.1

# Default run name; override with: sbatch finetune.sbatch --run-name focal-soundscape
RUN_NAME=${RUN_NAME:-finetune}

echo "=== EfficientNet-B1 SED with Soft Labels ==="
echo "Config: configs/finetune-sed.yaml"
echo "Run name: $RUN_NAME"

uv run --project $repo_dir/user/anthony birdclef-efficientnet-sed \
    --config configs/finetune-sed.yaml --run-name "$RUN_NAME" "$@"
