"""GPU-accelerated mel-spectrogram using nnAudio.

Drop-in replacement for MelSpectrogramTransform that runs on GPU.
Designed to be used inside the model's forward pass so mel computation
happens on-device without CPU<->GPU transfers.

Usage:
    mel_layer = NNAudioMelSpectrogram(sample_rate=32000)
    mel_layer = mel_layer.to("cuda")

    # In model forward pass:
    waveforms = waveforms.to("cuda")       # (B, num_samples)
    mel = mel_layer(waveforms)             # (B, 1, 128, 500)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features import MelSpectrogram


class NNAudioMelSpectrogram(nn.Module):
    """GPU-accelerated log-mel spectrogram with normalization.

    Uses nnAudio for GPU-native STFT and mel filterbank computation.
    Produces identical output shape and normalization as MelSpectrogramTransform.

    Args:
        sample_rate: Audio sample rate (default: 32000 Hz for BirdSet)
        n_mels: Number of mel frequency bins (default: 128)
        frame_shift_ms: Frame shift in milliseconds (default: 10ms)
        target_frames: Target number of time frames (default: 500 for ~5s)
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
        win_length = int(0.025 * sample_rate)  # 25ms window
        n_fft = 2 ** math.ceil(math.log2(win_length))

        self.mel_spec = MelSpectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=None,
            htk=True,  # Match torchaudio's mel_scale="htk"
            norm=None,  # Match torchaudio's norm=None (no area normalization)
            trainable_mel=False,
            trainable_STFT=False,
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

        # nnAudio returns (batch, n_mels, time) in linear power scale
        mel = self.mel_spec(waveform)

        # Log scale (match torchaudio: additive epsilon, not clamp)
        mel = torch.log(mel + 1e-6)

        # Pad or crop to target_frames
        n_frames = mel.shape[-1]
        if n_frames < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - n_frames))
        elif n_frames > self.target_frames:
            start = (n_frames - self.target_frames) // 2
            mel = mel[..., start : start + self.target_frames]

        # Normalize: (mel - mean) / (std * 2)
        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        # Add channel dim: (batch, n_mels, T) -> (batch, 1, n_mels, T)
        mel = mel.unsqueeze(-3)

        if single:
            mel = mel.squeeze(0)

        return mel
