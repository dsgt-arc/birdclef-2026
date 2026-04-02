#!/bin/bash
#SBATCH --job-name=103c-pretrain
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1200 -qinferno
#SBATCH --output=logs/pretrain-103c-%j.out

# 103c tiny MSE-only pretraining with auto-resume.
# Submits one follow-up job on completion to continue from checkpoint.
# ~1.7h/epoch, 50 epochs total, ~11 epochs per 20h job → 4-5 jobs total.

set -eu

CONFIG=configs/pretrain-103c-tiny-fp32.yaml
CKPT_DIR=logs/pretrain-103c/checkpoints

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

source "$exp_dir/sbatch/_setup-venv.sh"

echo "=== 103c Tiny SSL Pretraining ==="
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Resume from checkpoint if it exists
CKPT_ARGS=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
    echo "Resuming from $CKPT_DIR/last.ckpt"
    CKPT_ARGS="--ckpt-path last"
fi

birdclef-ssl-pretrain --config "$CONFIG" --ckpt-dir "$CKPT_DIR" $CKPT_ARGS
EXIT_CODE=$?

# If training finished (exit 0) but may not have reached max_epochs,
# check if last.ckpt exists and submit one follow-up job.
if [ $EXIT_CODE -eq 0 ] && [ -f "$CKPT_DIR/last.ckpt" ]; then
    # Check if training reached max_epochs or early stopped
    if grep -q "max_epochs=\|EarlyStopping" logs/pretrain-103c-${SLURM_JOB_ID}.out 2>/dev/null; then
        echo "Training finished (max_epochs or early stopping), no follow-up needed."
    else
        echo "Submitting follow-up job to continue training..."
        sbatch "$exp_dir/sbatch/pretrain-103c-chain.sbatch"
    fi
fi
