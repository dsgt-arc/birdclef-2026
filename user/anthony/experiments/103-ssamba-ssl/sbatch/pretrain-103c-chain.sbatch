#!/bin/bash
#SBATCH --job-name=103c-pretrain
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t1200 -qinferno
#SBATCH --output=logs/pretrain-103c-%j.out
#SBATCH --signal=B:USR1@120

# 103c tiny MSE-only pretraining with auto-resume.
# Submits one follow-up job on completion to continue from checkpoint.
# ~1.9h/epoch, 50 epochs total, ~10 epochs per 20h job → 5 jobs total.
#
# SLURM sends SIGUSR1 120s before wall limit. We trap it to submit
# the follow-up job before the process is killed.

set -eu

CONFIG=configs/pretrain-103c-tiny-fp32.yaml
CKPT_DIR=logs/pretrain-103c/checkpoints

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

REQUEUE=false

submit_followup() {
    if [ "$REQUEUE" = true ]; then
        return
    fi
    REQUEUE=true
    echo "Submitting follow-up job to continue training..."
    sbatch "$exp_dir/sbatch/pretrain-103c-chain.sbatch"
}

# Trap SIGUSR1 (sent 120s before timeout) to submit follow-up
trap 'submit_followup' USR1

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

# Run training (allow non-zero exit from timeout/signal)
set +e
birdclef-ssl-pretrain --config "$CONFIG" --ckpt-dir "$CKPT_DIR" $CKPT_ARGS &
wait $!
EXIT_CODE=$?
set -e

echo "Training exited with code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ] && [ -f "$CKPT_DIR/last.ckpt" ]; then
    # Check if training reached max_epochs or early stopped
    if grep -q "max_epochs=\|EarlyStopping" logs/pretrain-103c-${SLURM_JOB_ID}.out 2>/dev/null; then
        echo "Training finished (max_epochs or early stopping), no follow-up needed."
    else
        submit_followup
    fi
elif [ $EXIT_CODE -ne 0 ] && [ -f "$CKPT_DIR/last.ckpt" ]; then
    # Timeout or signal — submit follow-up if not already done
    submit_followup
fi
