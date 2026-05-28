"""Sydorskyi 2025 RandomFiltering (4-band random-EQ waveform augmentation) — direct port.

Source: vendor/sydorskyi-2025/code_base/models/augmentations.py @ 053594f

Applied to the WAVEFORM (pre-mel) by Sydorskyi's forward wrapper as `batch_aug`
(cls_forward.py:199). It performs a random per-band gain in the STFT domain:

    1. STFT (n_fft=1024, hop=512, hann; NOTE: different n_fft from the mel's 2048)
    2. sample n_bands gains from Uniform[min_db, 0] dB (per-sample, per-band)
    3. linear-interpolate the band gains to all freq bins
    4. dB → amplitude (10 ** (dB/20)), multiply STFT bins
    5. iSTFT
    6. (if normalize_wave) peak-normalize

Verified config (docs/sydorskyi-config-verified.md § RandomFiltering, selected_ebs.py:182):
    n_bands=4, min_db=-20, is_wave=True, normalize_wave=True (LATE_NORMALIZE),
    n_fft=1024, hop_length=512

Confidence: VERIFIED (Phase A.2). 2026-05-28.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RandomFiltering(nn.Module):
    """Random 4-band EQ applied to a waveform (B, T) via STFT → gain → iSTFT.

    Args:
        n_bands: number of random-gain control points (default 4).
        min_db: lower bound of the per-band gain (Uniform[min_db, 0]); Sydorskyi -20.
        is_wave: True = input is a waveform (do STFT/iSTFT roundtrip); False = input
            is already a complex spectrogram (B, bins, time) → only apply gain.
        normalize_wave: peak-normalize the filtered waveform afterwards.
        n_fft / hop_length: STFT params (default 1024 / 512 — NOT the mel's 2048).
        eps: peak-normalize denominator floor.
    """

    def __init__(
        self,
        n_bands: int = 4,
        min_db: float = -15,
        is_wave: bool = True,
        normalize_wave: bool = False,
        n_fft: int = 1024,
        hop_length: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.min_db = min_db
        self.is_wave = is_wave
        self.normalize_wave = normalize_wave
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer("hann_window", torch.hann_window(self.n_fft))

    def spec_forward(self, s: Tensor) -> Tensor:
        # s shape: (B, bins, time). Sample per-sample, per-band gains in dB.
        filter_points = torch.rand(s.shape[0], self.n_bands, device=s.device) * self.min_db
        # Interpolate band gains up to bins (interpolate needs a channel dim).
        filter_coeffs = nn.functional.interpolate(
            filter_points.unsqueeze(1), mode="linear", size=s.shape[1]
        ).squeeze(1)
        # dB → amplitude.
        filter_coeffs = 10 ** (filter_coeffs / 20)
        return s * filter_coeffs.unsqueeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_wave:
            complex_spec = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=None,
                window=self.hann_window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=None,
                return_complex=True,
            )
            filtered_spec = self.spec_forward(complex_spec)
            filtered_wave = torch.istft(
                filtered_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=None,
                window=self.hann_window,
                center=True,
                normalized=False,
                onesided=None,
                length=None,
                return_complex=False,
            )
            if self.normalize_wave:
                filtered_wave = filtered_wave / (
                    filtered_wave.abs().max(axis=1, keepdims=True)[0] + self.eps
                )
            return filtered_wave
        else:
            return self.spec_forward(x)
