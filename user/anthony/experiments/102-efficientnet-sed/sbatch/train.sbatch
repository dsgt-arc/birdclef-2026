#!/bin/bash
#SBATCH --job-name=102-train
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t720 -qinferno
#SBATCH --output=logs/train-%j.out

# Usage:
#   sbatch sbatch/train.sbatch configs/run-a-frozen.yaml
#   sbatch sbatch/train.sbatch configs/run-b-lora.yaml
#   sbatch sbatch/train.sbatch configs/run-b-lora.yaml --run-name lora-r16

set -eu

CONFIG=${1:?Usage: sbatch train.sbatch <config.yaml> [extra args]}
shift  # remaining args passed to CLI

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/102-efficientnet-sed

# Venv in TMPDIR (per CLAUDE.md policy)
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

module load cuda/12.1.1

uv pip install -e "$repo_dir/user/anthony/experiments/009-birdclef-pipeline"
uv pip install -e "$exp_dir"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd $exp_dir

echo "=== EfficientNet-B1 Training ==="
echo "Config: $CONFIG"
echo "Extra args: $@"

birdclef-efficientnet --config "$CONFIG" "$@"
