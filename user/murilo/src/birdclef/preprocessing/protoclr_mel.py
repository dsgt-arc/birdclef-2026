"""Mel spectrogram transform matching ProtoCLR's preprocessing pipeline.

Reproduces the exact preprocessing from ProtoCLR's MelSpectrogramProcessor:
torchaudio MelSpectrogram -> AmplitudeToDB -> min-max norm -> standardize.

Parameters matched to the Xeno-Canto-6s-16khz training set.
Reference: https://huggingface.co/ilyassmoummad/ProtoCLR/blob/main/melspectrogram.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms as T

# ProtoCLR training set statistics (Xeno-Canto)
PROTOCLR_MEAN = 0.5347
PROTOCLR_STD = 0.0772


class ProtoCLRMelSpectrogram(nn.Module):
    """Mel spectrogram matching ProtoCLR's preprocessing.

    Args:
        sample_rate: Audio sample rate (default: 16000 Hz)
        n_mels: Number of mel bins (default: 128)
        n_fft: FFT size (default: 1024)
        hop_length: Hop length in samples (default: 320)
        f_min: Min frequency in Hz (default: 50)
        f_max: Max frequency in Hz (default: 8000)
        target_frames: Target number of time frames (None = no padding/cropping)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        f_min: int = 50,
        f_max: int = 8000,
        target_frames: int | None = None,
    ):
        super().__init__()
        self.target_frames = target_frames

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to ProtoCLR-compatible mel spectrogram.

        Args:
            waveform: Audio tensor - single (samples,) or batched (batch, samples).

        Returns:
            Single input:  (1, n_mels, time)
            Batched input: (batch, 1, n_mels, time)
        """
        single = waveform.dim() == 1
        if single:
            waveform = waveform.unsqueeze(0)

        # Mel spectrogram + dB scale
        mel = self.mel_spec(waveform)  # (batch, n_mels, time)
        mel = self.amplitude_to_db(mel)

        # Min-max normalization (per-sample)
        B = mel.shape[0]
        mel_flat = mel.reshape(B, -1)
        mel_min = mel_flat.min(dim=-1, keepdim=True).values
        mel_max = mel_flat.max(dim=-1, keepdim=True).values
        mel_flat = (mel_flat - mel_min) / (mel_max - mel_min + 1e-8)
        mel = mel_flat.reshape(mel.shape)

        # Standardize with ProtoCLR training set stats
        mel = (mel - PROTOCLR_MEAN) / PROTOCLR_STD

        # Pad or crop to target_frames if specified
        if self.target_frames is not None:
            n_frames = mel.shape[-1]
            if n_frames < self.target_frames:
                mel = F.pad(mel, (0, self.target_frames - n_frames))
            elif n_frames > self.target_frames:
                mel = mel[..., : self.target_frames]

        # Add channel dim: (batch, n_mels, T) -> (batch, 1, n_mels, T)
        mel = mel.unsqueeze(1)

        if single:
            mel = mel.squeeze(0)

        return mel
