#!/bin/bash
#SBATCH --job-name=full-ssamba
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH -t480 -qembers
#SBATCH --output=logs/full-finetune-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Full finetuning SSAMBA on ESC-50"
echo "Job ID: $SLURM_JOB_ID"
echo "Fold: ${FOLD:-1}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony
gpu_stats_file=logs/gpu_stats-full-${SLURM_JOB_ID}.csv

# Start GPU monitoring in background (every 5 seconds)
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > "$gpu_stats_file" &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

# Sync venv to local NVMe for faster package loading
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/${SLURM_JOB_ID}/.venv

# Use copy mode to avoid hardlink failures between scratch and tmpdir filesystems
export UV_LINK_MODE=copy

# Set HuggingFace cache to scratch
export HF_HOME=$scratch_dir/.cache/huggingface

# Sync dependencies from user directory (where pyproject.toml lives)
cd $user_dir
echo "Installing with CUDA kernel support..."
# mamba-ssm and causal-conv1d wheels are sourced from GitHub releases in pyproject.toml
uv sync --extra cuda

# Diagnostic: test mamba-ssm CUDA kernel import
echo ""
echo "=== Testing mamba-ssm CUDA kernels ==="
${UV_PROJECT_ENVIRONMENT}/bin/python -c "
import sys
print(f'Python: {sys.executable}')
try:
    import mamba_ssm
    print(f'mamba_ssm version: {mamba_ssm.__version__}')
    print(f'mamba_ssm location: {mamba_ssm.__file__}')
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print('CUDA kernels: AVAILABLE')
except Exception as e:
    print(f'mamba_ssm import failed: {e}')
    sys.exit(1)
"
echo "=== End mamba-ssm test ==="
echo ""

cd $repo_dir

# Run full finetuning (matching original SSAMBA hyperparameters)
# FOLD: 1-5 (default: 1) - single fold for quick testing
# EPOCHS: 50 (default)
# LR: 1e-4 (default)
# BATCH_SIZE: 8 (default, smaller due to full param gradients)
# GRAD_ACCUM: 6 (default, effective batch = 8*6=48 to match original)
# TARGET_LENGTH: 512 (default, matches 5s ESC-50 audio without zero padding)
# FSTRIDE/TSTRIDE: 10 (default, overlapping patches matching original)
# DROP_PATH: 0.1 (default, matching original)
# NO_SPECAUGMENT: set to disable SpecAugment
# NO_NOISE_AUG: set to disable noise augmentation
${UV_PROJECT_ENVIRONMENT}/bin/python -m birdclef.finetune_full \
    --encoder ssamba-hf \
    --esc50-root $repo_dir/vendor/ESC-50 \
    --output-dir $scratch_dir/outputs/esc50-full \
    --checkpoint-dir $scratch_dir/checkpoints \
    --fold ${FOLD:-1} \
    --epochs ${EPOCHS:-50} \
    --batch-size ${BATCH_SIZE:-8} \
    --lr ${LR:-1e-4} \
    --max-grad-norm ${MAX_GRAD_NORM:-1.0} \
    --grad-accum-steps ${GRAD_ACCUM:-6} \
    --target-length ${TARGET_LENGTH:-512} \
    --fstride ${FSTRIDE:-10} \
    --tstride ${TSTRIDE:-10} \
    --drop-path-rate ${DROP_PATH:-0.1} \
    --lr-decay-start ${LR_DECAY_START:-6} \
    ${NO_SPECAUGMENT:+--no-specaugment} \
    ${NO_NOISE_AUG:+--no-noise-aug}

echo ""
echo "Done!"
echo "GPU stats saved to: $gpu_stats_file"
date
