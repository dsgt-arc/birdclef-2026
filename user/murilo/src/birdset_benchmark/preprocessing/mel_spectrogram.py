"""Mel-spectrogram generation for audio preprocessing.

Used by: Experiment 002 (Supervised ViT Baseline)

Following patterns from EAT encoder (encoders/eat.py) with Kaldi fbank generation.
"""

import torch
import torch.nn as nn
import torchaudio


class MelSpectrogramTransform(nn.Module):
    """On-the-fly Kaldi fbank generation with normalization.

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
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_shift = frame_shift_ms
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized mel-spectrogram.

        Args:
            waveform: Audio waveform tensor, shape (samples,) or (1, samples)

        Returns:
            Normalized mel-spectrogram, shape (1, n_mels, target_frames)
        """
        # Ensure 1D waveform
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # Compute Kaldi fbank (following EAT encoder patterns)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),  # Add batch dim for kaldi
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.n_mels,
            dither=0.0,
            frame_shift=self.frame_shift,
        )  # Shape: (num_frames, n_mels)

        # Transpose to (n_mels, num_frames)
        fbank = fbank.transpose(0, 1)

        # Pad or crop to target_frames
        if fbank.shape[1] < self.target_frames:
            # Pad with zeros
            pad_length = self.target_frames - fbank.shape[1]
            fbank = torch.nn.functional.pad(fbank, (0, pad_length), mode="constant", value=0)
        elif fbank.shape[1] > self.target_frames:
            # Crop from center
            start_idx = (fbank.shape[1] - self.target_frames) // 2
            fbank = fbank[:, start_idx : start_idx + self.target_frames]

        # Normalize: (mel - mean) / (std * 2)
        # Following EAT encoder normalization pattern
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # Add channel dimension: (1, n_mels, target_frames)
        fbank = fbank.unsqueeze(0)

        return fbank

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass for nn.Module compatibility."""
        return self(waveform)
