#!/bin/bash
#SBATCH --job-name=009-predict
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t120 -qinferno
#SBATCH --output=logs/predict-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

module load cuda/12.1.1

# Set checkpoint path (update after training)
CKPT_PATH=${1:-logs/finetune/checkpoints/last.ckpt}

uv run --project $repo_dir/user/anthony birdclef-predict \
    --config configs/finetune.yaml \
    --checkpoint $CKPT_PATH \
    --soundscape-dir $scratch_dir/data/birdclef-2025/test_soundscapes/ \
    --sample-submission $scratch_dir/data/birdclef-2025/sample_submission.csv \
    --output predictions/submission.csv \
    --device cuda
