#!/bin/bash
#SBATCH --job-name=lora-eat
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t240 -qembers
#SBATCH --output=logs/lora-finetune-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "LoRA finetuning on ESC-50"
echo "Job ID: $SLURM_JOB_ID"
echo "Encoder: ${ENCODER:-eat}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony
gpu_stats_file=logs/gpu_stats-lora-${SLURM_JOB_ID}.csv

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
if [[ "${ENCODER:-eat}" == "ssamba-hf" ]]; then
    echo "Installing with CUDA kernel support for ssamba-hf..."
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
else
    uv sync
fi

cd $repo_dir

# Run LoRA finetuning
# ENCODER: eat (default), wav2vec2, or ssamba-hf
# LORA_RANK: 8 (default) or 16
# LR: model-specific default if not set (eat=1e-4, wav2vec2=1e-5, ssamba-hf=1e-4)
# MAX_GRAD_NORM: 1.0 (default)
# FOLD: single fold to run (default: all 5)
# CHECKPOINT_DIR: required for ssamba-hf (default: $scratch_dir/checkpoints)
# TARGET_LENGTH/FSTRIDE/TSTRIDE/DROP_PATH: ssamba-hf specific (see finetune_lora.py)
${UV_PROJECT_ENVIRONMENT}/bin/python -m birdclef.finetune_lora \
    --encoder ${ENCODER:-eat} \
    --esc50-root $repo_dir/vendor/ESC-50 \
    --output-dir $scratch_dir/outputs/esc50-lora \
    --lora-rank ${LORA_RANK:-8} \
    --epochs ${EPOCHS:-50} \
    --batch-size ${BATCH_SIZE:-16} \
    --max-grad-norm ${MAX_GRAD_NORM:-1.0} \
    ${FOLD:+--fold $FOLD} \
    ${LR:+--lr $LR} \
    ${CHECKPOINT_DIR:+--checkpoint-dir $CHECKPOINT_DIR} \
    ${TARGET_LENGTH:+--target-length $TARGET_LENGTH} \
    ${FSTRIDE:+--fstride $FSTRIDE} \
    ${TSTRIDE:+--tstride $TSTRIDE} \
    ${DROP_PATH:+--drop-path-rate $DROP_PATH} \
    ${NO_NOISE_AUG:+--no-noise-aug}

echo ""
echo "Done!"
echo "GPU stats saved to: $gpu_stats_file"
date
