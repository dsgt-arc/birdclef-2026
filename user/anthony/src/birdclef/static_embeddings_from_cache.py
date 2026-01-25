"""Generate static embeddings from teacher cache using model2vec methodology.

Alternative to the original approach that requires raw patch embeddings.
This method directly averages contextualized teacher embeddings per token ID,
then applies the model2vec pipeline (PCA + SIF weighting).

model2vec pipeline:
1. Build token embeddings by averaging contextualized embeddings per token
2. Apply PCA with whitening (removes common directions, reduces dimensionality)
3. Compute SIF/Zipf weights for weighted pooling at inference time

Original approach:
    patches → average by token → codebook → transformer → PCA → weights

This approach:
    teacher_embeds → average by token → PCA → weights

The teacher cache already contains (teacher_embeds, token_ids) pairs,
so we can compute static embeddings without running the encoder again.

Usage:
    python -m birdclef.static_embeddings_from_cache \
        --cache-dir /path/to/teacher_cache \
        --output-dir /path/to/output \
        --pca-dim 256 \
        --num-workers 8

References:
    - Model2Vec: https://github.com/MinishLab/model2vec
    - SIF weighting: https://openreview.net/pdf?id=SyK00v5xx
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ShardDataset(Dataset):
    """Dataset for loading teacher cache shards in parallel."""

    def __init__(self, shard_files: list[Path]):
        self.shard_files = shard_files

    def __len__(self) -> int:
        return len(self.shard_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a shard and return flattened tensors ready for accumulation."""
        try:
            tensors = load_file(str(self.shard_files[idx]))
            teacher_embeds = tensors["teacher_embeds"]  # (N, seq_len, 768)
            token_ids = tensors["token_ids"]  # (N, seq_len)

            # Flatten for accumulation
            flat_embeds = teacher_embeds.view(-1, teacher_embeds.shape[-1]).float()
            flat_ids = token_ids.view(-1).long()

            return flat_embeds, flat_ids
        except Exception as e:
            print(f"Error loading {self.shard_files[idx]}: {e}")
            # Return empty tensors on error
            return torch.zeros(0, 768), torch.zeros(0, dtype=torch.long)


def collate_shards(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate multiple shards by concatenating."""
    embeds = torch.cat([b[0] for b in batch if b[0].shape[0] > 0], dim=0)
    ids = torch.cat([b[1] for b in batch if b[1].shape[0] > 0], dim=0)
    return embeds, ids


def build_static_embeddings_from_cache(
    cache_dir: str | Path,
    num_tokens: int = 1024,
    embed_dim: int = 768,
    max_shards: int | None = None,
    num_workers: int = 8,
    prefetch_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build static embeddings by averaging teacher embeddings per token ID.

    Uses PyTorch DataLoader for parallel I/O and index_add_ for vectorized
    accumulation. This is ~3-4x faster than the sequential NumPy version.

    Args:
        cache_dir: Directory containing safetensors cache files
        num_tokens: Number of discrete tokens in vocabulary (BEATs uses 1024)
        embed_dim: Embedding dimension (768 for BEATs)
        max_shards: Maximum shards to process (None for all)
        num_workers: Number of parallel workers for data loading
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        Tuple of (static_embeddings, token_counts, stats):
        - static_embeddings: (num_tokens, embed_dim) array
        - token_counts: (num_tokens,) array of counts per token
        - stats: Dictionary with token distribution statistics
    """
    cache_dir = Path(cache_dir)
    shard_files = sorted(cache_dir.glob("*.safetensors"))

    if not shard_files:
        raise ValueError(f"No safetensors files found in {cache_dir}")

    if max_shards:
        shard_files = shard_files[:max_shards]

    print(f"Building static embeddings from {len(shard_files)} shards...")
    print(f"  Using {num_workers} workers with prefetch_factor={prefetch_factor}")

    # Create dataset and dataloader for parallel I/O
    dataset = ShardDataset(shard_files)
    loader = DataLoader(
        dataset,
        batch_size=1,  # One shard per batch
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_shards,
        persistent_workers=num_workers > 0,
    )

    # Accumulators - use float64 for numerical stability
    token_sums = torch.zeros(num_tokens, embed_dim, dtype=torch.float64)
    token_counts = torch.zeros(num_tokens, dtype=torch.long)

    total_patches = 0

    for flat_embeds, flat_ids in tqdm(loader, desc="Processing shards"):
        if flat_embeds.shape[0] == 0:
            continue

        n_patches = flat_embeds.shape[0]
        total_patches += n_patches

        # Vectorized accumulation using index_add_
        # This is O(N) instead of O(num_tokens * N)
        token_sums.index_add_(0, flat_ids, flat_embeds.double())
        token_counts.index_add_(
            0, flat_ids, torch.ones(n_patches, dtype=torch.long)
        )

    # Compute means
    static_embeddings = torch.zeros(num_tokens, embed_dim, dtype=torch.float32)
    mask = token_counts > 0
    static_embeddings[mask] = (token_sums[mask] / token_counts[mask].unsqueeze(1)).float()

    # Convert to numpy
    static_embeddings = static_embeddings.numpy()
    token_counts = token_counts.numpy()

    # Compute statistics
    used_tokens = mask.sum().item()
    stats = {
        "num_tokens": num_tokens,
        "used_tokens": int(used_tokens),
        "unused_tokens": int(num_tokens - used_tokens),
        "total_patches": int(total_patches),
        "token_counts": token_counts.tolist(),
        "min_count": int(token_counts[token_counts > 0].min()) if used_tokens > 0 else 0,
        "max_count": int(token_counts.max()),
        "mean_count": float(token_counts[token_counts > 0].mean()) if used_tokens > 0 else 0,
        "source": "teacher_cache",
        "method": "mean_contextualized_embeddings",
        "num_workers": num_workers,
    }

    print(f"\nRaw static embeddings computed:")
    print(f"  Shape: {static_embeddings.shape}")
    print(f"  Used tokens: {used_tokens}/{num_tokens}")
    print(f"  Total patches: {total_patches:,}")
    print(f"  Token counts: min={stats['min_count']}, max={stats['max_count']:,}, mean={stats['mean_count']:.1f}")

    return static_embeddings, token_counts, stats


def apply_pca(
    embeddings: np.ndarray,
    n_components: int = 256,
    whiten: bool = True,
) -> tuple[np.ndarray, object]:
    """Apply PCA dimensionality reduction (model2vec key step).

    PCA serves two purposes:
    1. Reduces dimensionality for faster inference
    2. Removes noise and common biases in the embedding space

    Whitening normalizes variance along each component, which helps
    with downstream tasks by making dimensions equally important.

    Args:
        embeddings: Input embeddings (num_tokens, embed_dim)
        n_components: Target dimensionality
        whiten: Whether to whiten the output (recommended for model2vec)

    Returns:
        Tuple of (reduced_embeddings, pca_model)
    """
    from sklearn.decomposition import PCA

    # Only fit on used tokens (non-zero embeddings)
    mask = np.any(embeddings != 0, axis=1)
    used_embeddings = embeddings[mask]

    n_components = min(n_components, used_embeddings.shape[0], used_embeddings.shape[1])

    print(f"\nApplying PCA:")
    print(f"  Input dim: {embeddings.shape[1]}")
    print(f"  Output dim: {n_components}")
    print(f"  Whitening: {whiten}")
    print(f"  Fitting on {used_embeddings.shape[0]} used tokens")

    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(used_embeddings)

    # Transform all embeddings
    reduced = pca.transform(embeddings)

    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    return reduced.astype(np.float32), pca


def compute_sif_weights(
    token_counts: np.ndarray,
    a: float = 1e-3,
) -> np.ndarray:
    """Compute Smooth Inverse Frequency (SIF) weights (model2vec key step).

    SIF weighting downweights common tokens and upweights rare ones.
    Formula: w = a / (a + p) where p is the token probability.

    This is similar to IDF in information retrieval and helps reduce
    the influence of very common (less informative) tokens.

    Args:
        token_counts: Count of each token (num_tokens,)
        a: Smoothing parameter (default 1e-3, lower = more aggressive downweighting)

    Returns:
        SIF weights (num_tokens,)
    """
    total = token_counts.sum()
    if total == 0:
        return np.ones_like(token_counts, dtype=np.float32)

    # Compute probabilities
    probs = token_counts / total

    # SIF formula: w = a / (a + p)
    weights = a / (a + probs)

    # Handle zero-count tokens (unused tokens get weight 0)
    weights[token_counts == 0] = 0.0

    print(f"\nSIF weights computed (a={a}):")
    print(f"  Range: [{weights[weights > 0].min():.4f}, {weights.max():.4f}]")
    print(f"  Mean: {weights[weights > 0].mean():.4f}")

    return weights.astype(np.float32)


def compute_zipf_weights(
    token_counts: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """Compute Zipf-based weights for negative sampling style weighting.

    Uses the same weighting as Word2Vec negative sampling.
    Formula: w = count^alpha / sum(count^alpha)

    Args:
        token_counts: Count of each token (num_tokens,)
        alpha: Exponent (0.75 is Word2Vec default, 0 = uniform, 1 = proportional)

    Returns:
        Zipf weights (num_tokens,)
    """
    powered = np.power(token_counts.astype(np.float64), alpha)
    total = powered.sum()
    if total == 0:
        return np.ones_like(token_counts, dtype=np.float32)

    weights = powered / total

    print(f"\nZipf weights computed (alpha={alpha}):")
    print(f"  Range: [{weights[weights > 0].min():.6f}, {weights.max():.6f}]")

    return weights.astype(np.float32)


def save_artifacts(
    output_dir: str | Path,
    static_embeddings: np.ndarray,
    static_embeddings_pca: np.ndarray,
    token_counts: np.ndarray,
    stats: dict,
    pca_model: object,
    sif_weights: np.ndarray,
    zipf_weights: np.ndarray | None = None,
) -> None:
    """Save all model2vec artifacts.

    Args:
        output_dir: Output directory
        static_embeddings: Full-dim static embeddings (num_tokens, 768)
        static_embeddings_pca: PCA-reduced embeddings (num_tokens, pca_dim)
        token_counts: Token counts for reference
        stats: Statistics dictionary
        pca_model: Fitted PCA model
        sif_weights: SIF weights per token
        zipf_weights: Optional Zipf weights per token
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_dir / "static_embeddings.npy", static_embeddings)
    np.save(output_dir / "static_embeddings_pca.npy", static_embeddings_pca)
    np.save(output_dir / "token_counts.npy", token_counts)
    np.save(output_dir / "sif_weights.npy", sif_weights)

    if zipf_weights is not None:
        np.save(output_dir / "zipf_weights.npy", zipf_weights)

    # Save PCA model
    with open(output_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca_model, f)

    # Save statistics
    with open(output_dir / "token_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nArtifacts saved to {output_dir}:")
    print(f"  static_embeddings.npy: {static_embeddings.shape}")
    print(f"  static_embeddings_pca.npy: {static_embeddings_pca.shape}")
    print(f"  token_counts.npy: {token_counts.shape}")
    print(f"  sif_weights.npy: {sif_weights.shape}")
    if zipf_weights is not None:
        print(f"  zipf_weights.npy: {zipf_weights.shape}")
    print(f"  pca_model.pkl")
    print(f"  token_stats.json")


def build_model2vec_embeddings(
    cache_dir: str | Path,
    output_dir: str | Path,
    num_tokens: int = 1024,
    pca_dim: int = 256,
    sif_a: float = 1e-3,
    max_shards: int | None = None,
    num_workers: int = 8,
) -> dict:
    """Full model2vec pipeline: build static embeddings with PCA and weighting.

    Args:
        cache_dir: Directory containing safetensors cache files
        output_dir: Output directory for artifacts
        num_tokens: Number of discrete tokens (1024 for BEATs)
        pca_dim: PCA output dimensionality (256 is typical for model2vec)
        sif_a: SIF smoothing parameter
        max_shards: Maximum shards to process (None for all)
        num_workers: Number of parallel workers for data loading

    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("MODEL2VEC STATIC EMBEDDINGS FROM TEACHER CACHE")
    print("=" * 60)

    # Step 1: Build raw static embeddings
    static_embeddings, token_counts, stats = build_static_embeddings_from_cache(
        cache_dir=cache_dir,
        num_tokens=num_tokens,
        max_shards=max_shards,
        num_workers=num_workers,
    )

    # Step 2: Apply PCA with whitening
    static_embeddings_pca, pca_model = apply_pca(
        static_embeddings,
        n_components=pca_dim,
        whiten=True,
    )
    stats["pca_dim"] = pca_dim
    stats["pca_explained_variance"] = float(pca_model.explained_variance_ratio_.sum())

    # Step 3: Compute SIF weights
    sif_weights = compute_sif_weights(token_counts, a=sif_a)
    stats["sif_a"] = sif_a

    # Step 4: Compute Zipf weights (alternative weighting)
    zipf_weights = compute_zipf_weights(token_counts, alpha=0.75)

    # Save all artifacts
    save_artifacts(
        output_dir=output_dir,
        static_embeddings=static_embeddings,
        static_embeddings_pca=static_embeddings_pca,
        token_counts=token_counts,
        stats=stats,
        pca_model=pca_model,
        sif_weights=sif_weights,
        zipf_weights=zipf_weights,
    )

    print("\n" + "=" * 60)
    print("MODEL2VEC PIPELINE COMPLETE")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate model2vec static embeddings from teacher cache"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory containing safetensors cache files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for static embeddings",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum shards to process (default: all)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1024,
        help="Number of discrete tokens (default: 1024 for BEATs)",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=256,
        help="PCA output dimensionality (default: 256)",
    )
    parser.add_argument(
        "--sif-a",
        type=float,
        default=1e-3,
        help="SIF smoothing parameter (default: 1e-3)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for data loading (default: 8)",
    )

    args = parser.parse_args()

    build_model2vec_embeddings(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        num_tokens=args.num_tokens,
        pca_dim=args.pca_dim,
        sif_a=args.sif_a,
        max_shards=args.max_shards,
        num_workers=args.num_workers,
    )
