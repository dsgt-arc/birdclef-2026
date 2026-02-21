"""Mel-spectrogram generation for audio preprocessing.

Vectorized implementation using torchaudio.transforms.MelSpectrogram
for efficient batched processing in DataLoader collate functions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelSpectrogramTransform(nn.Module):
    """Vectorized log-mel spectrogram with normalization.

    Supports both single waveforms and batched input for efficient
    collate-time processing without per-sample Python loops.

    Args:
        sample_rate: Audio sample rate (default: 32000 Hz for BirdSet)
        n_mels: Number of mel frequency bins (default: 128)
        frame_shift_ms: Frame shift in milliseconds (default: 10ms)
        target_frames: Target number of time frames, pads/crops if needed (default: 500 for ~5s)
        norm_mean: Normalization mean (default: AudioSet mean=-4.268)
        norm_std: Normalization std (default: AudioSet std=4.569)
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 500,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        super().__init__()
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        hop_length = int(frame_shift_ms / 1000.0 * sample_rate)
        win_length = int(0.025 * sample_rate)  # 25ms window (standard)
        n_fft = 2 ** math.ceil(math.log2(win_length))  # next power of 2 for FFT efficiency

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform(s) to normalized log-mel spectrogram(s).

        Args:
            waveform: Audio tensor — single (samples,) or batched (batch, samples).

        Returns:
            Single input:  (1, n_mels, target_frames)
            Batched input: (batch, 1, n_mels, target_frames)
        """
        single = waveform.dim() == 1
        if single:
            waveform = waveform.unsqueeze(0)

        # Compute mel spectrogram for entire batch at once: (batch, n_mels, time)
        mel = self.mel_spec(waveform)

        # Log scale (add epsilon to avoid log(0))
        mel = torch.log(mel + 1e-6)

        # Pad or crop to target_frames
        n_frames = mel.shape[-1]
        if n_frames < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - n_frames))
        elif n_frames > self.target_frames:
            start = (n_frames - self.target_frames) // 2
            mel = mel[..., start : start + self.target_frames]

        # Normalize: (mel - mean) / (std * 2)
        # Following EAT encoder normalization pattern
        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        # Add channel dim: (..., 1, n_mels, target_frames)
        mel = mel.unsqueeze(-3)

        if single:
            mel = mel.squeeze(0)

        return mel
