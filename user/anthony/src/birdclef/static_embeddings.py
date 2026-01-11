"""Static embeddings distillation for BEATs.

Implements Tokenizer-guided model2vec approach:
1. Extract patch embeddings from encoder frontend
2. Extract token IDs from tokenizer
3. Build codebook by averaging patches per token
4. Apply PCA for dimensionality reduction (model2vec key step)
5. Apply Zipf/SIF weighting (model2vec key step)
6. Compute static embeddings by passing centroids through transformer

References:
- Model2Vec: https://github.com/MinishLab/model2vec
- BEATs Tokenizer: https://arxiv.org/abs/2212.09058
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add vendor path for BEATs imports
VENDOR_PATH = Path(__file__).parents[4] / "vendor" / "unilm" / "beats"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))


def load_beats_encoder(checkpoint_path: str | Path) -> nn.Module:
    """Load BEATs encoder model."""
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_beats_tokenizer(checkpoint_path: str | Path) -> nn.Module:
    """Load BEATs tokenizer model."""
    from Tokenizers import Tokenizers, TokenizersConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TokenizersConfig(checkpoint["cfg"])
    model = Tokenizers(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def extract_patch_embeddings(
    encoder: nn.Module,
    waveform: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract patch embeddings from encoder frontend (before transformer).

    Args:
        encoder: BEATs encoder model
        waveform: Audio waveform (batch, [1,] samples) at 16kHz
        device: Device to run on

    Returns:
        Patch embeddings (batch, num_patches, 768)
    """
    encoder = encoder.to(device)

    # Handle (batch, 1, samples) -> (batch, samples)
    if waveform.dim() == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)

    waveform = waveform.to(device)

    with torch.no_grad():
        # Compute fbank features - preprocess expects (batch, samples)
        fbank = encoder.preprocess(waveform)
        fbank = fbank.unsqueeze(1)  # (batch, 1, time, 128)

        # Patch embedding
        features = encoder.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)

        # Layer norm
        features = encoder.layer_norm(features)

        # Project to encoder dim
        if encoder.post_extract_proj is not None:
            features = encoder.post_extract_proj(features)

    return features.cpu()


def extract_token_ids(
    tokenizer: nn.Module,
    waveform: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract discrete token IDs from tokenizer.

    Args:
        tokenizer: BEATs tokenizer model
        waveform: Audio waveform (batch, [1,] samples) at 16kHz
        device: Device to run on

    Returns:
        Token IDs (batch, num_patches)
    """
    tokenizer = tokenizer.to(device)

    # Handle (batch, 1, samples) -> (batch, samples)
    if waveform.dim() == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)

    waveform = waveform.to(device)

    with torch.no_grad():
        token_ids = tokenizer.extract_labels(waveform)

    return token_ids.cpu()


def extract_all_patches_and_tokens(
    encoder: nn.Module,
    tokenizer: nn.Module,
    dataloader,
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract all patch embeddings and token IDs from a dataset.

    Args:
        encoder: BEATs encoder model
        tokenizer: BEATs tokenizer model
        dataloader: DataLoader yielding (waveform, label) tuples
        device: Device to run on
        max_samples: Maximum number of samples to process (None for all)

    Returns:
        Tuple of (all_patches, all_tokens):
        - all_patches: (total_patches, 768)
        - all_tokens: (total_patches,)
    """
    all_patches = []
    all_tokens = []
    n_samples = 0

    for waveforms, _ in tqdm(dataloader, desc="Extracting patches"):
        # Extract patch embeddings
        patches = extract_patch_embeddings(encoder, waveforms, device)
        # Extract token IDs
        tokens = extract_token_ids(tokenizer, waveforms, device)

        # Flatten batch and patch dimensions
        batch_size, num_patches, embed_dim = patches.shape
        patches_flat = patches.reshape(-1, embed_dim)
        tokens_flat = tokens.reshape(-1)

        all_patches.append(patches_flat.numpy())
        all_tokens.append(tokens_flat.numpy())

        n_samples += batch_size
        if max_samples and n_samples >= max_samples:
            break

    return np.concatenate(all_patches), np.concatenate(all_tokens)


def build_codebook(
    patches: np.ndarray,
    tokens: np.ndarray,
    num_tokens: int = 1024,
) -> tuple[np.ndarray, dict]:
    """Build codebook by averaging patches per token ID.

    Args:
        patches: All patch embeddings (N, 768)
        tokens: Token IDs for each patch (N,)
        num_tokens: Number of discrete tokens in vocabulary

    Returns:
        Tuple of (codebook, stats):
        - codebook: (num_tokens, 768) - averaged patch embeddings per token
        - stats: Dictionary with token distribution statistics
    """
    embed_dim = patches.shape[1]
    codebook = np.zeros((num_tokens, embed_dim), dtype=np.float32)
    token_counts = np.zeros(num_tokens, dtype=np.int64)

    # Accumulate patches per token
    for i in range(num_tokens):
        mask = tokens == i
        count = mask.sum()
        token_counts[i] = count
        if count > 0:
            codebook[i] = patches[mask].mean(axis=0)

    # Stats
    used_tokens = (token_counts > 0).sum()
    stats = {
        "num_tokens": num_tokens,
        "used_tokens": int(used_tokens),
        "unused_tokens": int(num_tokens - used_tokens),
        "total_patches": int(len(tokens)),
        "token_counts": token_counts.tolist(),
        "min_count": int(token_counts[token_counts > 0].min()) if used_tokens > 0 else 0,
        "max_count": int(token_counts.max()),
        "mean_count": float(token_counts[token_counts > 0].mean()) if used_tokens > 0 else 0,
    }

    return codebook, stats


def compute_static_embeddings(
    encoder: nn.Module,
    codebook: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute static embeddings by passing codebook through transformer.

    Each codebook entry is treated as a single-patch sequence.

    Args:
        encoder: BEATs encoder model
        codebook: Codebook centroids (num_tokens, 768)
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        Static embeddings (num_tokens, 768)
    """
    encoder = encoder.to(device)
    encoder.eval()

    num_tokens, embed_dim = codebook.shape
    static_embeddings = np.zeros((num_tokens, embed_dim), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, num_tokens, batch_size), desc="Computing static embeddings"):
            end = min(start + batch_size, num_tokens)
            batch = torch.tensor(codebook[start:end], dtype=torch.float32, device=device)

            # Add sequence dimension: (batch, 1, 768) - single patch per sequence
            batch = batch.unsqueeze(1)

            # Pass through transformer encoder
            x, _ = encoder.encoder(batch, padding_mask=None)

            # Mean pool (trivial for single patch)
            x = x.mean(dim=1)

            static_embeddings[start:end] = x.cpu().numpy()

    return static_embeddings


def apply_pca(
    embeddings: np.ndarray,
    n_components: int = 256,
    whiten: bool = True,
) -> tuple[np.ndarray, PCA]:
    """Apply PCA dimensionality reduction (model2vec key step).

    PCA serves two purposes:
    1. Reduces dimensionality for faster inference
    2. Removes noise and common biases in the embedding space

    Args:
        embeddings: Input embeddings (num_tokens, embed_dim)
        n_components: Target dimensionality
        whiten: Whether to whiten the output (recommended)

    Returns:
        Tuple of (reduced_embeddings, pca_model)
    """
    # Only fit on used tokens (non-zero embeddings)
    mask = np.any(embeddings != 0, axis=1)
    used_embeddings = embeddings[mask]

    pca = PCA(n_components=min(n_components, used_embeddings.shape[0], used_embeddings.shape[1]), whiten=whiten)
    pca.fit(used_embeddings)

    # Transform all embeddings
    reduced = pca.transform(embeddings)

    return reduced, pca


def compute_sif_weights(
    token_counts: np.ndarray,
    a: float = 1e-3,
) -> np.ndarray:
    """Compute Smooth Inverse Frequency (SIF) weights (model2vec key step).

    SIF weighting downweights common tokens and upweights rare ones.
    Formula: w = a / (a + p) where p is the token probability.

    This is similar to IDF in information retrieval.

    Args:
        token_counts: Count of each token (num_tokens,)
        a: Smoothing parameter (default 1e-3)

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

    # Handle zero-count tokens
    weights[token_counts == 0] = 0.0

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
    return weights.astype(np.float32)


def save_artifacts(
    output_dir: str | Path,
    codebook: np.ndarray,
    static_embeddings: np.ndarray,
    stats: dict,
    pca_model: Optional[PCA] = None,
    sif_weights: Optional[np.ndarray] = None,
    static_embeddings_pca: Optional[np.ndarray] = None,
):
    """Save all distillation artifacts.

    Args:
        output_dir: Output directory
        codebook: Codebook centroids (num_tokens, 768)
        static_embeddings: Static embeddings (num_tokens, 768)
        stats: Token distribution statistics
        pca_model: Fitted PCA model (optional)
        sif_weights: SIF weights per token (optional)
        static_embeddings_pca: PCA-reduced static embeddings (optional)
    """
    import json
    import pickle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "codebook.npy", codebook)
    np.save(output_dir / "static_embeddings.npy", static_embeddings)

    with open(output_dir / "token_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved artifacts to {output_dir}")
    print(f"  - codebook.npy: {codebook.shape}")
    print(f"  - static_embeddings.npy: {static_embeddings.shape}")
    print(f"  - token_stats.json")

    if pca_model is not None:
        with open(output_dir / "pca_model.pkl", "wb") as f:
            pickle.dump(pca_model, f)
        print(f"  - pca_model.pkl")

    if sif_weights is not None:
        np.save(output_dir / "sif_weights.npy", sif_weights)
        print(f"  - sif_weights.npy: {sif_weights.shape}")

    if static_embeddings_pca is not None:
        np.save(output_dir / "static_embeddings_pca.npy", static_embeddings_pca)
        print(f"  - static_embeddings_pca.npy: {static_embeddings_pca.shape}")
