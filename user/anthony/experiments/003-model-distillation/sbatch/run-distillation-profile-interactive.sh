#!/bin/bash
#SBATCH --job-name=distill-profile
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH -t60 -qinferno
#SBATCH -p gpu-rtx6000
#SBATCH --output=logs/distill-profile-%j.out
set -eu

# Usage: sbatch run-distillation-profile-interactive.sh
#        sbatch --export=LIMIT_BATCHES=200,BATCH_SIZE=64 run-distillation-profile-interactive.sh
#
# Quick profiling run for distillation training performance analysis.
# Uses v2 tensor format with static embeddings (symmetric with run-distillation-500k.sbatch).
# Runs limited batches with PyTorch profiler enabled to identify bottlenecks.

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Configuration with defaults (tuned for profiling)
STUDENT=${STUDENT:-convnext-small}
BATCH_SIZE=${BATCH_SIZE:-64}
LIMIT_BATCHES=${LIMIT_BATCHES:-100}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-20}
NUM_WORKERS=${NUM_WORKERS:-6}
LR=${LR:-1e-4}

PROFILE_ID=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Distillation Profiling Run"
echo "=============================================="
echo "Profile ID: ${PROFILE_ID}"
echo "Student model: ${STUDENT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Limit batches: ${LIMIT_BATCHES} train, ${LIMIT_VAL_BATCHES} val"
echo "Num workers: ${NUM_WORKERS}"
echo "Learning rate: ${LR}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony

# Sync venv to local NVMe for faster package loading
# Use --extra dev to get py-spy for profiling
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/${SLURM_JOB_ID}/.venv
cd $user_dir
uv sync --extra dev

cd $repo_dir

# Setup cache directories (v2 tensor format with static embeddings)
source_cache_dir=$scratch_dir/outputs/exp003-distillation/teacher_cache_500k_v2
static_embeddings=$scratch_dir/outputs/exp003-distillation/static_embeddings_500k_v2/static_embeddings.npy
local_cache_dir=${TMPDIR}/teacher_cache_profile
output_dir=$scratch_dir/outputs/exp003-profile-${PROFILE_ID}

mkdir -p "$output_dir"

echo ""
echo "Cache setup:"
echo "  Source cache: $source_cache_dir ($(du -sh $source_cache_dir 2>/dev/null | cut -f1))"
echo "  Static embeddings: $static_embeddings"
echo "  Local cache: $local_cache_dir"
echo "  Output dir: $output_dir"

# Run training with PyTorch profiler only
# (py-spy removed - too slow with subprocesses, PyTorch profiler is sufficient)
echo ""
echo "Starting training with PyTorch profiler..."

${UV_PROJECT_ENVIRONMENT}/bin/python \
    $user_dir/experiments/003-model-distillation/sbatch/train_lightning.py \
    --student "$STUDENT" \
    --source-cache "$source_cache_dir" \
    --static-embeddings "$static_embeddings" \
    --local-cache "$local_cache_dir" \
    --output-dir "$output_dir" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --max-epochs 1 \
    --limit-train-batches "$LIMIT_BATCHES" \
    --limit-val-batches "$LIMIT_VAL_BATCHES" \
    --patience 999 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor 4 \
    --shards-per-worker 3 \
    --experiment-name "profile-${PROFILE_ID}" \
    --precision "32" \
    --profile \
    --profile-dir "$output_dir/profiler"

echo ""
echo "=============================================="
echo "Profiling complete!"
echo "=============================================="
echo "Profile ID: ${PROFILE_ID}"
echo "PyTorch Profiler: $output_dir/profiler"
echo "TensorBoard logs: $output_dir"
echo ""
echo "To view traces:"
echo "  tensorboard --logdir $output_dir"
echo "  # Or open profiler/*.json in chrome://tracing"
date
