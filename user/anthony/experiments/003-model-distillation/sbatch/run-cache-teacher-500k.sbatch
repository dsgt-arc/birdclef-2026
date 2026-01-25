#!/bin/bash
#SBATCH --job-name=cache-teacher-500k
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/cache-teacher-500k-%A_%a.out
#SBATCH --array=0-14

# Usage: sbatch run-cache-teacher-500k.sbatch
#
# Processes AudioSet-500k shards in parallel using SLURM array jobs.
# Each array task handles ~10 shards (145 total / 15 tasks = ~10 shards each).
# Uses v2 cache format (no static_embeds - 50% space savings).
#
# Idempotent: skips already cached shards.
#
# To process all 145 shards with 15 parallel jobs:
#   sbatch --array=0-14 run-cache-teacher-500k.sbatch
#
# To rerun failed tasks (e.g., tasks 3 and 7):
#   sbatch --array=3,7 run-cache-teacher-500k.sbatch

set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Configuration
SHARDS_PER_TASK=10
TOTAL_SHARDS=145
DATASET_VERSION=${DATASET_VERSION:-500k}

echo "=============================================="
echo "Teacher Caching Array Job - v2 format"
echo "=============================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Shards per task: ${SHARDS_PER_TASK}"
echo "Dataset: ${DATASET_VERSION}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony

# Sync venv to local NVMe for faster package loading
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/.venv
export HF_HOME=$scratch_dir/.cache/huggingface
cd $user_dir
uv sync

# Paths
audioset_dir=$scratch_dir/data/audioset-wds/${DATASET_VERSION}
cache_dir=$scratch_dir/outputs/exp003-distillation/teacher_cache_${DATASET_VERSION}_v2
encoder_path=$scratch_dir/checkpoints/beats/BEATs_iter3.pt
tokenizer_path=$scratch_dir/checkpoints/beats/Tokenizers_iter3.pt
local_tar_dir=${TMPDIR:-/tmp}/audioset_shards_${SLURM_ARRAY_TASK_ID}

# Calculate shard range for this task
START_SHARD=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
END_SHARD=$((START_SHARD + SHARDS_PER_TASK - 1))
if [ $END_SHARD -ge $TOTAL_SHARDS ]; then
    END_SHARD=$((TOTAL_SHARDS - 1))
fi

echo ""
echo "This task processes shards ${START_SHARD} to ${END_SHARD}"

# Copy relevant shards to local NVMe for faster IO
echo ""
echo "Copying shards to local NVMe..."
mkdir -p $local_tar_dir
mkdir -p $cache_dir

for i in $(seq $START_SHARD $END_SHARD); do
    shard_name=$(printf "shard-%05d.tar" $i)
    if [ -f "$audioset_dir/train/$shard_name" ]; then
        # Check if already cached
        cache_file="$cache_dir/$(printf "shard-%05d.safetensors" $i)"
        if [ -f "$cache_file" ]; then
            echo "  Skipping $shard_name (already cached)"
        else
            echo "  Copying $shard_name"
            cp "$audioset_dir/train/$shard_name" "$local_tar_dir/"
        fi
    fi
done

echo ""
echo "Local tar directory:"
ls -lh $local_tar_dir/ 2>/dev/null | head -5 || echo "  (empty - all shards already cached)"

cd $repo_dir

${UV_PROJECT_ENVIRONMENT}/bin/python -c "
from pathlib import Path
from birdclef.caching import TeacherCacher
from safetensors.torch import save_file
from tqdm import tqdm

local_tar_dir = Path('$local_tar_dir')
cache_dir = Path('$cache_dir')
encoder_path = Path('$encoder_path')
tokenizer_path = Path('$tokenizer_path')

# Find tar files to process
tar_files = sorted(local_tar_dir.glob('shard-*.tar'))
print(f'Found {len(tar_files)} tar files to process')

if len(tar_files) == 0:
    print('All shards already cached - nothing to do!')
else:
    cacher = TeacherCacher(
        encoder_path=encoder_path,
        tokenizer_path=tokenizer_path,
    )
    stats = cacher.cache_shards(tar_files, cache_dir)
    print(f'Processed {stats[\"processed\"]} shards ({stats[\"total_samples\"]} samples)')
"

echo ""
echo "Array task ${SLURM_ARRAY_TASK_ID} complete!"
date
