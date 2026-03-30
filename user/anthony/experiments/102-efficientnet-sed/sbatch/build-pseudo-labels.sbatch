#!/bin/bash
#SBATCH --job-name=102-pseudo
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH --array=0-49
#SBATCH -t60 -qinferno
#SBATCH --output=logs/pseudo-labels-%A_%a.out

# CPU-only array job: each task processes a shard of unlabeled soundscapes.
# 50 tasks × ~212 files each = 10,592 files total.
# Each task takes ~15-20 min on CPU.
#
# After all tasks complete, merge shards:
#   python scripts/merge_pseudo_labels.py

set -eu

exp_dir=$(realpath ~/arc/birdclef-2026/user/anthony/experiments/102-efficientnet-sed)
cd $exp_dir

# Use the scratch venv (has onnxruntime, librosa, etc.)
venv=~/scratch/birdclef-2026/.venv-010
source $venv/bin/activate

# Install onnxruntime if not present
pip install onnxruntime --quiet 2>/dev/null || true

ONNX_DIR="artifacts/weights/lora-attention"
N_SHARDS=50

echo "=== Pseudo-label shard $SLURM_ARRAY_TASK_ID / $N_SHARDS ==="

python scripts/build_soundscape_pseudo_labels.py \
    --onnx-dir "$ONNX_DIR" \
    --output-dir artifacts/pseudo_labels/parts/ \
    --n-shards $N_SHARDS \
    --shard-id $SLURM_ARRAY_TASK_ID
