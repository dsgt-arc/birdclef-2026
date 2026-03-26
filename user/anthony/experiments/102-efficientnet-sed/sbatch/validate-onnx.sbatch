#!/bin/bash
#SBATCH --job-name=102-validate
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t30 -qinferno
#SBATCH --output=logs/validate-%j.out

# Usage:
#   sbatch sbatch/validate-onnx.sbatch <checkpoint> <config> <onnx-dir>

set -eu

CHECKPOINT=${1:?Usage: sbatch validate-onnx.sbatch <checkpoint> <config> <onnx-dir>}
CONFIG=${2:?}
ONNX_DIR=${3:?}

exp_dir=$(realpath ~/arc/birdclef-2026/user/anthony/experiments/102-efficientnet-sed)

JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
mkdir -p "$(dirname $JOB_VENV)"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

module load cuda/12.1.1

uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install onnx onnxruntime librosa
uv pip install -e "$exp_dir"

cd $exp_dir

python scripts/validate_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --onnx-dir "$ONNX_DIR" \
    --n-samples 10
