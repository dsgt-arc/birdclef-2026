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

from birdclef.preprocessing.pcen import PCENLayer


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
        n_fft: FFT window size. If None, derived from 25ms window (default: None → 1024)
        hop_length: Hop length in samples. If None, derived from frame_shift_ms (default: None)
        fmax: Maximum frequency for mel filterbank. None = Nyquist (default: None)
        norm_type: Normalization type:
            - "audioset": log(mel+eps) → (x - mean) / (std * 2). Default, matches v1/v4a.
            - "amplitude_to_db": AmplitudeToDB(top_db=80) → per-sample standardize → min-max [0,1].
              Used by Tom's 28 experiments and 2025 1st place.
            - "pcen": PCEN normalization → per-sample min-max [0,1]. Adaptive gain
              control for recording-invariant features.
            - "pcen_amp2db": PCEN → AmplitudeToDB → standardize → min-max [0,1].
              PCEN adaptive gain control followed by dB compression.
        top_db: Dynamic range clipping for amplitude_to_db norm (default: 80.0).
        pcen_s: PCEN smoothing coefficient (default: 0.025).
        pcen_alpha: PCEN AGC strength (default: 0.98).
        pcen_delta: PCEN bias (default: 2.0).
        pcen_r: PCEN compression exponent (default: 0.5).
        pcen_trainable: Whether PCEN parameters are learnable (default: True).
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 500,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        n_fft: int | None = None,
        hop_length: int | None = None,
        fmax: int | None = None,
        norm_type: str = "audioset",
        top_db: float = 80.0,
        pcen_s: float = 0.025,
        pcen_alpha: float = 0.98,
        pcen_delta: float = 2.0,
        pcen_r: float = 0.5,
        pcen_trainable: bool = True,
    ):
        super().__init__()
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.norm_type = norm_type
        self.top_db = top_db

        if hop_length is None:
            hop_length = int(frame_shift_ms / 1000.0 * sample_rate)
        if n_fft is None:
            win_length = int(0.025 * sample_rate)  # 25ms window
            n_fft = 2 ** math.ceil(math.log2(win_length))
        # win_length = n_fft for explicit n_fft (standard STFT convention)
        win_length = n_fft

        self.mel_spec = MelSpectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=fmax,
            htk=True,  # Match torchaudio's mel_scale="htk"
            norm=None,  # Match torchaudio's norm=None (no area normalization)
            trainable_mel=False,
            trainable_STFT=False,
        )

        # PCEN layer (only created for pcen/pcen_amp2db norm types)
        self.pcen_layer = None
        if norm_type in ("pcen", "pcen_amp2db"):
            self.pcen_layer = PCENLayer(
                n_mels=n_mels,
                s=pcen_s,
                alpha=pcen_alpha,
                delta=pcen_delta,
                r=pcen_r,
                trainable=pcen_trainable,
            )

    def _normalize_audioset(self, mel: torch.Tensor) -> torch.Tensor:
        """AudioSet log normalization: log(mel+eps) → (x - mean) / (std * 2)."""
        mel = torch.log(mel + 1e-6)
        return (mel - self.norm_mean) / (self.norm_std * 2)

    def _normalize_amplitude_to_db(self, mel: torch.Tensor) -> torch.Tensor:
        """AmplitudeToDB → per-sample standardize → min-max [0,1].

        Matches Tom's experiments and 2025 1st place normalization pipeline.
        mel input is in linear power scale from nnAudio.
        """
        # Power-to-dB: 10 * log10(mel + eps), clipped to top_db range
        mel = 10.0 * torch.log10(mel + 1e-10)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = torch.clamp(mel, min=mel_max - self.top_db)

        # Per-sample standardize (zero mean, unit variance)
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        mel = (mel - mean) / std

        # Per-sample min-max scale to [0, 1]
        mel_min = mel.amin(dim=(-2, -1), keepdim=True)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)

        return mel

    def _normalize_pcen(self, mel: torch.Tensor) -> torch.Tensor:
        """PCEN → per-sample min-max [0,1].

        mel input is in linear power scale from nnAudio.
        """
        mel = self.pcen_layer(mel)

        # Per-sample min-max scale to [0, 1]
        mel_min = mel.amin(dim=(-2, -1), keepdim=True)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)

        return mel

    def _normalize_simple_range(self, mel: torch.Tensor) -> torch.Tensor:
        """Simple range normalization: (X + 80) / 80.

        Maps AmplitudeToDB output from [-80, 0] to [0, 1].
        Used by Salman Ahmed (BirdCLEF 2025, 0.908 LB). Simpler than
        global mean/std — no dataset statistics needed.
        """
        # Power-to-dB: 10 * log10(mel + eps), clipped to top_db range
        mel = 10.0 * torch.log10(mel + 1e-10)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = torch.clamp(mel, min=mel_max - self.top_db)

        # Simple range mapping: [-80, 0] → [0, 1]
        mel = (mel + self.top_db) / self.top_db

        return mel

    def _normalize_pcen_amp2db(self, mel: torch.Tensor) -> torch.Tensor:
        """PCEN → AmplitudeToDB → standardize → min-max [0,1].

        Applies PCEN adaptive gain control on linear mel, then the standard
        AmplitudeToDB pipeline. PCEN output is ≥ 0, so log10 is valid.
        """
        mel = self.pcen_layer(mel)

        # AmplitudeToDB on PCEN output (≥ 0)
        mel = 10.0 * torch.log10(mel + 1e-10)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = torch.clamp(mel, min=mel_max - self.top_db)

        # Per-sample standardize
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        mel = (mel - mean) / std

        # Per-sample min-max scale to [0, 1]
        mel_min = mel.amin(dim=(-2, -1), keepdim=True)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)

        return mel

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

        # Pad or crop to target_frames (before normalization for consistent shape)
        n_frames = mel.shape[-1]
        if n_frames < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - n_frames))
        elif n_frames > self.target_frames:
            start = (n_frames - self.target_frames) // 2
            mel = mel[..., start : start + self.target_frames]

        # Normalize
        if self.norm_type == "amplitude_to_db":
            mel = self._normalize_amplitude_to_db(mel)
        elif self.norm_type == "simple_range":
            mel = self._normalize_simple_range(mel)
        elif self.norm_type == "pcen":
            mel = self._normalize_pcen(mel)
        elif self.norm_type == "pcen_amp2db":
            mel = self._normalize_pcen_amp2db(mel)
        else:
            mel = self._normalize_audioset(mel)

        # Add channel dim: (batch, n_mels, T) -> (batch, 1, n_mels, T)
        mel = mel.unsqueeze(-3)

        if single:
            mel = mel.squeeze(0)

        return mel
