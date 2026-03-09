#!/bin/bash
#SBATCH -J exp016-zero-baseline
#SBATCH -A paceship-clef2026_birdclef
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH -t 01:30:00
#SBATCH --mem=48G

set -euo pipefail
module load cuda/12.1.1

EXPERIMENT_DIR="$HOME/arc/birdclef-2026/user/anthony/experiments/016-contrastive-sed"
cd "$EXPERIMENT_DIR"
mkdir -p logs

# Redirect output after cd so paths are correct
exec > "logs/zero-training-${SLURM_JOB_ID}.out" 2> "logs/zero-training-${SLURM_JOB_ID}.err"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Experiment dir: $EXPERIMENT_DIR"
echo "TMPDIR: $TMPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- venv setup in TMPDIR ---
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"
mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

# Install PyTorch + dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers nnAudio soundfile scikit-learn pandas numpy pytorch-lightning

# Install birdclef_009 for data loading
PIPELINE_DIR="$EXPERIMENT_DIR/../009-birdclef-pipeline"
uv pip install -e "$PIPELINE_DIR"

echo "=== Running zero-training baseline ==="

# Optional: pass exp 012 checkpoint for strategy C
EXP012_CKPT="${1:-}"
EXTRA_ARGS=""
if [ -n "$EXP012_CKPT" ] && [ -f "$EXP012_CKPT" ]; then
    EXTRA_ARGS="--exp012-ckpt $EXP012_CKPT"
    echo "Using exp 012 checkpoint: $EXP012_CKPT"
fi

python scripts/zero_training_baseline.py \
    --batch-size 128 \
    --num-workers 4 \
    $EXTRA_ARGS

echo "=== Done ==="
