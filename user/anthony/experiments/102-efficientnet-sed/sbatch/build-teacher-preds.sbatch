#!/bin/bash
#SBATCH --job-name=102-teacher
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t240 -qinferno
#SBATCH --output=logs/teacher-preds-%j.out

# Runs the best attention model on all 244K focal chunks to generate
# per-chunk teacher predictions for data cleaning.

set -eu

exp_dir=$(realpath ~/arc/birdclef-2026/user/anthony/experiments/102-efficientnet-sed)

JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

module load cuda/12.1.1

uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -e "$exp_dir"

cd $exp_dir

CHECKPOINT="logs/lora-attention/checkpoints/lora-attention-epoch=01-val/cmap=0.5308.ckpt"
CONFIG="configs/run-i-efficientnet-attention.yaml"

echo "=== Teacher Predictions ==="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"

python scripts/build_teacher_predictions.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output artifacts/teacher_predictions.parquet \
    --batch-size 64
