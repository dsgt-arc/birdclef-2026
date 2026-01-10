"""Audio loading and preprocessing utilities."""

import torch
import torchaudio
import soundfile as sf
from pathlib import Path


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> torch.Tensor:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 16kHz for most models)
        mono: Convert to mono if True

    Returns:
        Audio tensor of shape (1, num_samples) if mono, else (channels, num_samples)
    """
    # Use soundfile directly to avoid torchaudio torchcodec issues
    data, sr = sf.read(path)
    waveform = torch.from_numpy(data).float()

    # Ensure shape is (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  # (samples, channels) -> (channels, samples)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad or trim waveform to target length.

    Args:
        waveform: Audio tensor of shape (channels, num_samples)
        target_length: Target number of samples

    Returns:
        Padded/trimmed tensor of shape (channels, target_length)
    """
    length = waveform.shape[-1]

    if length < target_length:
        # Pad with zeros
        padding = target_length - length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif length > target_length:
        # Trim
        waveform = waveform[..., :target_length]

    return waveform
