#!/bin/bash
#SBATCH --job-name=cache-teacher
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/cache-teacher-%j.out

# Usage: sbatch run-cache-teacher.sbatch
#
# Caches BEATs teacher embeddings and token IDs to safetensors shards.
# Uses v2 cache format (no static_embeds - 50% space savings).

set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "=============================================="
echo "Teacher Caching - v2 format"
echo "=============================================="
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Configuration
DATASET_VERSION=${DATASET_VERSION:-20k}
audioset_dir=$scratch_dir/data/audioset-wds/${DATASET_VERSION}
cache_dir=$scratch_dir/outputs/exp003-distillation/teacher_cache_${DATASET_VERSION}_v2
encoder_path=$scratch_dir/checkpoints/beats/BEATs_iter3.pt
tokenizer_path=$scratch_dir/checkpoints/beats/Tokenizers_iter3.pt

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_VERSION"
echo "  AudioSet dir: $audioset_dir"
echo "  Cache dir: $cache_dir"
echo ""

cd $repo_dir

uv run python -c "
from pathlib import Path
from birdclef.caching import cache_teacher_outputs

cache_teacher_outputs(
    audioset_dir=Path('$audioset_dir'),
    cache_dir=Path('$cache_dir'),
    encoder_path=Path('$encoder_path'),
    tokenizer_path=Path('$tokenizer_path'),
)
"

echo ""
echo "Teacher caching complete!"
date
