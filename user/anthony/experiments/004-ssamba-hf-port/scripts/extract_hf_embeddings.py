"""Extract embeddings using the HuggingFace-based SSAMBA port.

This implementation uses pure PyTorch and does not require CUDA.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np


def create_test_spectrogram(seed: int = 42) -> torch.Tensor:
    """Create a deterministic test spectrogram.

    Returns shape (batch, time_frames, freq_bins) = (1, 1024, 128)
    """
    rng = np.random.default_rng(seed)
    # Simulate mel spectrogram characteristics
    spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    return torch.from_numpy(spec)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings with HF SSAMBA port")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SSAMBA checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for embeddings")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()

    # Import from local package
    from ssamba_hf import SSAMBAModel, SSAMBAConfig, load_from_ssamba_checkpoint

    print(f"Loading model from {args.checkpoint}")
    model, config = load_from_ssamba_checkpoint(args.checkpoint, device=args.device)
    model = model.eval()

    print(f"Model config: {config}")

    print("Creating test spectrogram")
    spectrogram = create_test_spectrogram()
    spectrogram = spectrogram.to(args.device)

    print("Extracting embeddings")
    with torch.no_grad():
        embeddings = model.get_embeddings(spectrogram)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saving to {args.output}")

    # Save embeddings and metadata
    torch.save({
        "embeddings": embeddings.cpu(),
        "spectrogram": spectrogram.cpu(),
        "checkpoint": args.checkpoint,
    }, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
