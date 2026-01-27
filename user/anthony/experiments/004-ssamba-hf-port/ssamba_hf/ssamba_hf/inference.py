"""Long audio inference utilities for SSAMBA.

This module provides windowed inference for processing audio files longer than
the model's fixed input size (1024 frames). Since SSAMBA uses bidirectional
Mamba, true streaming inference is not possible - we use overlapping windows
and aggregate the results.

Usage:
    from ssamba_hf import SSAMBAModel, SSAMBAConfig
    from ssamba_hf.inference import get_embeddings_long

    model = SSAMBAModel(SSAMBAConfig.small())
    # Long spectrogram with 3000 frames
    spectrogram = torch.randn(1, 3000, 128)
    embeddings = get_embeddings_long(model, spectrogram)
"""

from typing import Literal, Optional, List
import torch
import torch.nn as nn


def get_embeddings_long(
    model: nn.Module,
    spectrogram: torch.Tensor,
    window_size: int = 1024,
    hop_size: int = 512,
    aggregation: Literal["mean", "max", "last"] = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Extract embeddings from long audio using windowed inference.

    Splits long spectrogram into overlapping windows, processes each through
    the model, and aggregates the results. This is necessary because:
    1. Bidirectional Mamba requires full sequence for forward+backward passes
    2. Learned positional embeddings are fixed to window_size

    Args:
        model: SSAMBA model instance
        spectrogram: Input spectrogram of shape (batch, time_frames, freq_bins)
            Can be any length >= window_size
        window_size: Size of each processing window (default: 1024, matching
            SSAMBA's expected input size)
        hop_size: Number of frames to advance between windows (default: 512,
            giving 50% overlap)
        aggregation: How to combine embeddings from multiple windows:
            - "mean": Average all window embeddings (default, most stable)
            - "max": Element-wise maximum across windows
            - "last": Use only the last window's embedding
        normalize: If True, L2-normalize the final embedding

    Returns:
        Embeddings tensor of shape (batch, embed_dim)

    Raises:
        ValueError: If spectrogram is shorter than window_size

    Example:
        >>> model = SSAMBAModel(SSAMBAConfig.small())
        >>> spec = torch.randn(1, 3000, 128)  # ~30 seconds of audio
        >>> emb = get_embeddings_long(model, spec, aggregation="mean")
        >>> emb.shape
        torch.Size([1, 384])
    """
    batch_size, num_frames, freq_bins = spectrogram.shape

    if num_frames < window_size:
        raise ValueError(
            f"Spectrogram has {num_frames} frames, but window_size is {window_size}. "
            f"Use model.get_embeddings() directly for short audio, or pad the input."
        )

    # If spectrogram is exactly window_size, just process directly
    if num_frames == window_size:
        return model.get_embeddings(spectrogram, normalize=normalize)

    # Calculate window positions
    window_starts = _compute_window_positions(num_frames, window_size, hop_size)

    # Process each window
    window_embeddings: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for start in window_starts:
            end = start + window_size
            window = spectrogram[:, start:end, :]
            emb = model.get_embeddings(window, normalize=False)
            window_embeddings.append(emb)

    # Stack embeddings: (num_windows, batch, embed_dim)
    stacked = torch.stack(window_embeddings, dim=0)

    # Aggregate
    if aggregation == "mean":
        result = stacked.mean(dim=0)
    elif aggregation == "max":
        result = stacked.max(dim=0).values
    elif aggregation == "last":
        result = stacked[-1]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Normalize if requested
    if normalize:
        result = nn.functional.normalize(result, p=2, dim=-1)

    return result


def _compute_window_positions(
    num_frames: int,
    window_size: int,
    hop_size: int,
) -> List[int]:
    """Compute starting positions for windows.

    Ensures the last window captures the end of the sequence, even if
    that means the final hop is smaller than hop_size.

    Args:
        num_frames: Total number of frames in the spectrogram
        window_size: Size of each window
        hop_size: Hop size between windows

    Returns:
        List of starting frame indices for each window
    """
    positions = []
    start = 0

    while start + window_size <= num_frames:
        positions.append(start)
        start += hop_size

    # Ensure we cover the end of the sequence
    final_start = num_frames - window_size
    if not positions or positions[-1] < final_start:
        positions.append(final_start)

    return positions


def estimate_num_windows(
    num_frames: int,
    window_size: int = 1024,
    hop_size: int = 512,
) -> int:
    """Estimate the number of windows for a given input length.

    Useful for memory/time estimation before processing.

    Args:
        num_frames: Total number of frames in the spectrogram
        window_size: Size of each window
        hop_size: Hop size between windows

    Returns:
        Number of windows that will be processed

    Example:
        >>> estimate_num_windows(3000, window_size=1024, hop_size=512)
        5
    """
    if num_frames < window_size:
        return 0
    if num_frames == window_size:
        return 1

    return len(_compute_window_positions(num_frames, window_size, hop_size))


def pad_to_window_size(
    spectrogram: torch.Tensor,
    window_size: int = 1024,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Pad a short spectrogram to minimum window size.

    Args:
        spectrogram: Input spectrogram of shape (batch, time_frames, freq_bins)
        window_size: Target minimum length
        pad_mode: Padding mode - "replicate", "reflect", or "constant"

    Returns:
        Padded spectrogram of shape (batch, max(time_frames, window_size), freq_bins)

    Example:
        >>> spec = torch.randn(1, 500, 128)
        >>> padded = pad_to_window_size(spec, window_size=1024)
        >>> padded.shape
        torch.Size([1, 1024, 128])
    """
    batch_size, num_frames, freq_bins = spectrogram.shape

    if num_frames >= window_size:
        return spectrogram

    pad_amount = window_size - num_frames

    # torch.nn.functional.pad expects (left, right) for last dim, then next-to-last, etc.
    # For shape (B, T, F), we want to pad T (dim 1), so we pad on dims 2 and 1:
    # pad = (0, 0, 0, pad_amount) means: no pad on last dim, pad_amount on right of second-to-last

    if pad_mode == "constant":
        # Pad with zeros
        padded = nn.functional.pad(spectrogram, (0, 0, 0, pad_amount), mode="constant", value=0)
    elif pad_mode == "replicate":
        # Replicate edge values - need to transpose for F.pad
        spec_t = spectrogram.transpose(1, 2)  # (B, F, T)
        padded_t = nn.functional.pad(spec_t, (0, pad_amount), mode="replicate")
        padded = padded_t.transpose(1, 2)  # (B, T, F)
    elif pad_mode == "reflect":
        spec_t = spectrogram.transpose(1, 2)  # (B, F, T)
        # Reflect mode requires pad_amount < num_frames
        if pad_amount >= num_frames:
            # Fall back to replicate for very short sequences
            padded_t = nn.functional.pad(spec_t, (0, pad_amount), mode="replicate")
        else:
            padded_t = nn.functional.pad(spec_t, (0, pad_amount), mode="reflect")
        padded = padded_t.transpose(1, 2)  # (B, T, F)
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    return padded
