#!/bin/bash
#SBATCH --job-name=102-export
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t30 -qinferno
#SBATCH --output=logs/export-%j.out

# Usage:
#   sbatch sbatch/export-onnx.sbatch logs/frozen/checkpoints/best.ckpt configs/run-a-frozen.yaml frozen
#   sbatch sbatch/export-onnx.sbatch logs/lora/checkpoints/best.ckpt configs/run-b-lora.yaml lora

set -eu

CHECKPOINT=${1:?Usage: sbatch export-onnx.sbatch <checkpoint> <config> <run-name>}
CONFIG=${2:?}
RUN_NAME=${3:?}

exp_dir=$(realpath ~/arc/birdclef-2026/user/anthony/experiments/102-efficientnet-sed)

# Venv in TMPDIR
JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
mkdir -p "$(dirname $JOB_VENV)"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

module load cuda/12.1.1

uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install onnx
uv pip install -e "$exp_dir"

cd $exp_dir

echo "=== ONNX Export: $RUN_NAME ==="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"

python scripts/export_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output-dir "artifacts/weights/$RUN_NAME/"
