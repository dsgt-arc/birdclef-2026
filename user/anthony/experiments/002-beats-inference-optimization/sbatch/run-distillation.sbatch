#!/bin/bash
#SBATCH --job-name=distill-002
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t120 -qembers
#SBATCH --output=logs/distillation-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Running BEATs static embeddings distillation"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Download ESC-50 if not present
esc50_dir=$scratch_dir/data/esc50
if [ ! -d "$esc50_dir/audio" ]; then
    echo "Downloading ESC-50 dataset..."
    mkdir -p $scratch_dir/data
    cd $scratch_dir/data

    # Download and extract
    wget -q https://github.com/karoldvl/ESC-50/archive/master.zip -O esc50.zip
    unzip -q esc50.zip
    mv ESC-50-master esc50
    rm esc50.zip

    echo "ESC-50 downloaded to $esc50_dir"
fi

cd $repo_dir

uv run python << 'EOF'
"""Static embeddings distillation for BEATs."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.static_embeddings import (
    load_beats_encoder,
    load_beats_tokenizer,
    extract_all_patches_and_tokens,
    build_codebook,
    compute_static_embeddings,
    apply_pca,
    compute_sif_weights,
    save_artifacts,
)
from birdclef.data.esc50 import ESC50Dataset

# Paths
scratch_dir = Path(os.path.expanduser("~/scratch/birdclef-2026"))
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
esc50_dir = scratch_dir / "data/esc50"
output_dir = scratch_dir / "outputs/exp002-benchmarks/static_embeddings"

device = "cpu"  # CPU-only for this job

print("=" * 60)
print("STATIC EMBEDDINGS DISTILLATION")
print("=" * 60)

# Load models
print("\nLoading models...")
encoder = load_beats_encoder(encoder_path)
tokenizer = load_beats_tokenizer(tokenizer_path)
print(f"  Encoder loaded from {encoder_path}")
print(f"  Tokenizer loaded from {tokenizer_path}")

# Load ESC-50 dataset (all folds)
print("\nLoading ESC-50 dataset...")
dataset = ESC50Dataset(esc50_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
print(f"  {len(dataset)} samples, {len(dataloader)} batches")

# Extract all patches and tokens
print("\nExtracting patches and tokens from ESC-50...")
all_patches, all_tokens = extract_all_patches_and_tokens(
    encoder, tokenizer, dataloader, device=device
)
print(f"  Extracted {len(all_patches)} patches")
print(f"  Patch shape: {all_patches.shape}")
print(f"  Token range: [{all_tokens.min()}, {all_tokens.max()}]")

# Build codebook
print("\nBuilding codebook...")
codebook, stats = build_codebook(all_patches, all_tokens, num_tokens=1024)
print(f"  Codebook shape: {codebook.shape}")
print(f"  Used tokens: {stats['used_tokens']}/{stats['num_tokens']}")
print(f"  Token counts: min={stats['min_count']}, max={stats['max_count']}, mean={stats['mean_count']:.1f}")

# Compute static embeddings
print("\nComputing static embeddings...")
static_embeddings = compute_static_embeddings(encoder, codebook, device=device, batch_size=64)
print(f"  Static embeddings shape: {static_embeddings.shape}")

# Apply PCA (model2vec key step)
print("\nApplying PCA dimensionality reduction...")
pca_dim = 256  # Common choice for model2vec
static_embeddings_pca, pca_model = apply_pca(static_embeddings, n_components=pca_dim)
print(f"  PCA reduced shape: {static_embeddings_pca.shape}")
print(f"  Explained variance ratio: {pca_model.explained_variance_ratio_.sum():.3f}")

# Compute SIF weights (model2vec key step)
print("\nComputing SIF weights...")
token_counts = np.array(stats["token_counts"])
sif_weights = compute_sif_weights(token_counts)
print(f"  SIF weights range: [{sif_weights[sif_weights > 0].min():.4f}, {sif_weights.max():.4f}]")

# Save artifacts
print("\nSaving artifacts...")
save_artifacts(
    output_dir, codebook, static_embeddings, stats,
    pca_model=pca_model, sif_weights=sif_weights, static_embeddings_pca=static_embeddings_pca
)

# Additional analysis: token distribution histogram
print("\n" + "=" * 60)
print("TOKEN DISTRIBUTION ANALYSIS")
print("=" * 60)
token_counts = np.array(stats["token_counts"])
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("\nToken count percentiles:")
for p in percentiles:
    val = np.percentile(token_counts[token_counts > 0], p)
    print(f"  {p}th percentile: {val:.0f}")

# Top 10 most common tokens
top_indices = np.argsort(token_counts)[-10:][::-1]
print("\nTop 10 most common tokens:")
for idx in top_indices:
    print(f"  Token {idx}: {token_counts[idx]} patches")

print("\nDistillation complete!")
EOF

echo ""
echo "Distillation complete!"
date
