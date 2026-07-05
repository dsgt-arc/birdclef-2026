#!/bin/bash
#SBATCH --job-name=distill-500k
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH -p gpu-rtx6000
#SBATCH --output=logs/distill-500k-%j.out
set -eu

# Usage: sbatch --export=STUDENT=convnext-small,MAX_EPOCHS=20,PATIENCE=3 run-distillation-500k.sbatch
#
# Distillation on AudioSet-500k using PyTorch Lightning
# - Per-worker shard affinity with aggressive prefetching
# - 143 shards, ~500k samples, 687GB cache
# - TensorBoard logging with system/data monitoring

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Configuration with defaults
STUDENT=${STUDENT:-convnext-small}
MAX_EPOCHS=${MAX_EPOCHS:-20}
PATIENCE=${PATIENCE:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
PROFILE=${PROFILE:-false}

echo "=============================================="
echo "Distillation - PyTorch Lightning"
echo "=============================================="
echo "Student model: ${STUDENT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Early stopping patience: ${PATIENCE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Profile: ${PROFILE}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony

# Sync venv to local NVMe for faster package loading
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/${SLURM_JOB_ID}/.venv
cd $user_dir
uv sync

cd $repo_dir

# Root-level GPU monitoring helper was removed during repo cleanup.
GPU_MONITOR_PID=""

# Cleanup function for background processes
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "${GPU_MONITOR_PID:-}" ]; then
        echo "  Stopping GPU monitor (PID: $GPU_MONITOR_PID)"
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Setup cache directories
source_cache_dir=$scratch_dir/outputs/exp003-distillation/teacher_cache_500k_v2
static_embeddings=$scratch_dir/outputs/exp003-distillation/static_embeddings_500k_v2/static_embeddings.npy
local_cache_dir=${TMPDIR}/teacher_cache_500k
output_dir=$scratch_dir/outputs/exp003-distillation-lightning

echo ""
echo "Cache setup:"
echo "  Source cache: $source_cache_dir ($(du -sh $source_cache_dir 2>/dev/null | cut -f1))"
echo "  Static embeddings: $static_embeddings"
echo "  Local cache: $local_cache_dir"
echo "  Output dir: $output_dir"

# Build profile flag
PROFILE_FLAG=""
if [ "$PROFILE" = "true" ]; then
    PROFILE_FLAG="--profile --profile-dir $output_dir/profiler"
fi

# Run training
${UV_PROJECT_ENVIRONMENT}/bin/python \
    $user_dir/experiments/003-model-distillation/sbatch/train_lightning.py \
    --student "$STUDENT" \
    --source-cache "$source_cache_dir" \
    --static-embeddings "$static_embeddings" \
    --local-cache "$local_cache_dir" \
    --output-dir "$output_dir" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --max-epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --num-workers 8 \
    --prefetch-factor 4 \
    --shards-per-worker 3 \
    --experiment-name "distillation-500k" \
    --precision "32" \
    $PROFILE_FLAG

echo ""
echo "Distillation complete for ${STUDENT}!"
echo "GPU utilization log: logs/gpu-${SLURM_JOB_ID}.csv"
echo "TensorBoard logs: $output_dir/distillation-500k/${STUDENT}"
date
