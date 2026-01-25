#!/bin/bash
#SBATCH --job-name=build-static-emb
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=16G
#SBATCH -t60 -qembers
#SBATCH --output=logs/build-static-embeddings-%j.out

# Usage: sbatch run-build-static-embeddings.sbatch
#
# Builds model2vec static embeddings from teacher cache.
# This generates static_embeddings.npy with full pipeline:
# 1. Average teacher embeddings per token ID
# 2. Apply PCA with whitening (768 -> 256 dims)
# 3. Compute SIF weights for weighted pooling
#
# Output files:
#   static_embeddings.npy      - Full 768-dim embeddings
#   static_embeddings_pca.npy  - PCA-reduced 256-dim embeddings
#   sif_weights.npy            - SIF weights per token
#   zipf_weights.npy           - Zipf weights per token
#   pca_model.pkl              - Fitted PCA model
#   token_stats.json           - Token distribution statistics

set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "=============================================="
echo "Building Model2Vec Static Embeddings"
echo "=============================================="
hostname
date

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Configuration
CACHE_DIR=${CACHE_DIR:-$scratch_dir/outputs/exp003-distillation/teacher_cache_500k}
OUTPUT_DIR=${OUTPUT_DIR:-$scratch_dir/outputs/exp003-distillation/static_embeddings_500k}
PCA_DIM=${PCA_DIM:-256}
MAX_SHARDS=${MAX_SHARDS:-}  # Empty = all shards

echo ""
echo "Configuration:"
echo "  Cache dir: $CACHE_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  PCA dim: $PCA_DIM"
echo "  Max shards: ${MAX_SHARDS:-all}"
echo ""

# Count shards
n_shards=$(ls -1 $CACHE_DIR/*.safetensors 2>/dev/null | wc -l)
echo "Found $n_shards shards in cache"
echo ""

cd $repo_dir

# Build max_shards argument
MAX_SHARDS_ARG=""
if [ -n "$MAX_SHARDS" ]; then
    MAX_SHARDS_ARG="--max-shards $MAX_SHARDS"
fi

uv run python -m birdclef.static_embeddings_from_cache \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pca-dim "$PCA_DIM" \
    $MAX_SHARDS_ARG

echo ""
echo "Static embeddings built successfully!"
echo "Output: $OUTPUT_DIR"
ls -lh $OUTPUT_DIR/
date
