"""Sydorskyi 2025 mel-spectrogram pipeline — direct port.

Source: vendor/sydorskyi-2025/code_base/models/blocks.py @ 053594f

Chain (when exportable=True, fixed_amplitude_to_db=True — matches selected_ebs.py / selected_eca.py):

    waveform (B, T_audio)
        → TraceableMelspec(sr=32000, n_mels=128, n_fft=2048, hop=512, f_min=20, normalized=True)
            → nnAudio STFT (window="hann", freq_scale="no", output_format="Complex")
            → power (|spec|^2, with normalized=True scaling by ||hann_window||_2)
            → torchaudio.MelScale (mel_scale="htk")
        → mel power spectrogram (B, n_mels, T_frame)
        → ChannelAgnosticAmplitudeToDB(top_db=80, amin=1e-10)
        → mel log spectrogram (B, n_mels, T_frame)
        → NormalizeMelSpec(exportable=True)
        → standardize (per-sample, over (n_mels, T_frame))
        → min-max[0, 1] (per-sample, over (n_mels, T_frame))
        → normalized mel (B, n_mels, T_frame)

Then WaveCNNAttenClasifier unsqueezes channel dim → (B, 1, n_mels, T_frame) → backbone.

For the BirdCLEF 2026 fine-tune we keep the SAME mel params as Sydorskyi
(matching the pretrained ckpt's training-time mel exactly). See
`docs/sydorskyi-config-verified.md` for the verified parameter table.

This module ports the minimum-necessary subset of blocks.py; we drop:
- QuantizableSTFT / QuantizableAmplitudeToDB (quantizable=True branch — unused)
- AmplitudeToDB (non-channel-agnostic variant — unused with fixed_amplitude_to_db=True)
- AttnBlock (atten_smoothing_config — unused)

Confidence: VERIFIED (Phase A.2 — every parameter line-cited in
docs/sydorskyi-config-verified.md). 2026-05-28.
"""

import math

import torch
import torch.nn as nn
import torchaudio.functional as F_audio
from nnAudio.Spectrogram import STFT as nnAudioSTFT
from torch import Tensor
from torchaudio.transforms import MelScale


class Swish(nn.Module):
    """Swish activation. Defensive port; not used by AttHead but kept for parity with blocks.py."""

    EPSILON_FP16 = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(x)
        gate = torch.clamp(gate, min=self.EPSILON_FP16, max=1.0 - self.EPSILON_FP16)
        return x * gate


def gem_freq(x: Tensor, p: float = 3.0, eps: float = 1e-6, exportable: bool = False) -> Tensor:
    """GeM (Generalized Mean) pooling over the FREQUENCY axis (dim=2 of NCHW tensor).

    GeM(x; p) = (mean(x^p)) ^ (1/p), with x clamped to ≥ eps for numerical safety.
    p → ∞ approaches max-pool; p = 1 approaches avg-pool; p = 3 is the Sydorskyi default.

    Args:
        x: backbone features (B, C, H_freq, W_time)
        p: GeM power exponent (default 3.0, learnable in GeMFreq module)
        eps: numerical safety floor
        exportable: when True, use a Tensor-only path that traces cleanly for ONNX export

    Returns:
        (B, C, 1, W_time)
    """
    if exportable:
        return x.clamp(min=eps).pow(p).mean(2, keepdims=True).pow(1.0 / p)
    else:
        return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    """Frequency-axis GeM pooling with a LEARNABLE p parameter.

    Source: vendor/sydorskyi-2025/code_base/models/blocks.py:71-79.

    Initialized at p=3.0; p is registered as nn.Parameter so it adapts during training.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, exportable: bool = False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.exportable = exportable

    def forward(self, x: Tensor) -> Tensor:
        return gem_freq(x, p=self.p, eps=self.eps, exportable=self.exportable)


class NormalizeMelSpec(nn.Module):
    """Per-sample standardize → min-max[0, 1] over the (n_mels, time) axes of a 3D mel tensor.

    Source: vendor/sydorskyi-2025/code_base/models/blocks.py:23-54.

    Standardize: z = (x - mean_over_(n_mels,time)) / (std_over_(n_mels,time) + eps)
    Then min-max[0, 1]: y = (z - amin_over_(n_mels,time)) / (amax_over_(n_mels,time) - amin + eps)

    The `exportable=True` branch (our default) is the simple amax/amin version — traces
    cleanly for ONNX. The non-exportable branch has a masked degenerate-input
    safeguard but does not trace.
    """

    def __init__(self, eps: float = 1e-6, exportable: bool = False):
        super().__init__()
        self.eps = eps
        self.exportable = exportable

    def forward(self, X: Tensor) -> Tensor:
        # Per-sample mean/std over (n_mels, time) — dims (1, 2) for NCT-like tensor.
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        if self.exportable:
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)
        else:
            # Non-exportable branch: handle degenerate samples (std=0 or identical values)
            # by leaving them as zeros instead of producing NaN.
            norm_min, norm_max = (
                Xstd.min(-1)[0].min(-1)[0],
                Xstd.max(-1)[0].max(-1)[0],
            )
            fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(norm_max - norm_min)
            V = torch.zeros_like(Xstd)
            if fix_ind.sum():
                V_fix = Xstd[fix_ind]
                norm_max_fix = norm_max[fix_ind, None, None]
                norm_min_fix = norm_min[fix_ind, None, None]
                V_fix = torch.max(
                    torch.min(V_fix, norm_max_fix),
                    norm_min_fix,
                )
                V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
                V[fix_ind] = V_fix
            return V


class ChannelAgnosticAmplitudeToDB(nn.Module):
    """Power/amplitude → decibel scale, accepting both 3D and 4D input.

    Source: vendor/sydorskyi-2025/code_base/models/blocks.py:822-880.

    Wraps `torchaudio.functional.amplitude_to_DB` so it accepts the (B, n_mels, time)
    tensor that comes out of TraceableMelspec.forward (3D) — the underlying torchaudio
    function expects (B, channel, n_mels, time). This wrapper adds and removes a fake
    channel dim as needed.

    Args:
        stype: "power" (default; multiplier=10) or "magnitude" (multiplier=20).
            Our mel chain feeds power (|STFT|^2), so stype="power" is correct.
        top_db: minimum-negative cutoff (default None; our config = 80).
        amin: numerical safety floor (default 1e-10).
    """

    __constants__ = ["multiplier", "ref_value", "db_multiplier"]

    def __init__(self, stype: str = "power", top_db: float | None = None, amin: float = 1e-10):
        super().__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = amin
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() in (3, 4), f"Expected 3D or 4D tensor, got {x.dim()}D"

        add_fake_channel = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            add_fake_channel = True

        x_db = F_audio.amplitude_to_DB(
            x, self.multiplier, self.amin, self.db_multiplier, self.top_db
        )

        if add_fake_channel:
            x_db = x_db.squeeze(1)

        return x_db


class TraceableMelspec(nn.Module):
    """ONNX-traceable mel spectrogram: nnAudio STFT + torchaudio MelScale.

    Source: vendor/sydorskyi-2025/code_base/models/blocks.py:448-524.

    Sydorskyi uses this hybrid (nnAudio STFT for ONNX-traceable 1D-conv internals,
    torchaudio MelScale for the standard htk mel filterbank) when `exportable=True`.
    Our selected_ebs.py / selected_eca.py configs both have exportable=True, so this
    is the path the pretrained ckpts were trained against.

    Important nuances vs torchaudio.transforms.MelSpectrogram:
    - The `normalized=True` flag here divides the complex STFT output by
      ||hann_window||_2 = sqrt(sum(hann²)) BEFORE the power calculation, which is
      slightly different from torchaudio.MelSpectrogram's `normalized=True`
      (which normalizes after power, by total window energy).
    - The mel filterbank itself is torchaudio's `MelScale` (htk-style).

    For Phase C Gate 1 parity, this implementation must match Sydorskyi's exactly
    — no shortcuts to plain torchaudio.MelSpectrogram.
    """

    def __init__(
        self,
        win_length: int | None = None,
        hop_length: int | None = None,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        # Mel params
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_fft: int = 400,
        norm: str | None = None,
        mel_scale: str = "htk",
        # Extra
        trainable: bool = False,
        **_,  # ignore extra args (e.g. quantizable=False from config)
    ):
        super().__init__()
        self.spectrogram = nnAudioSTFT(
            n_fft=n_fft,
            win_length=win_length,
            freq_bins=None,
            hop_length=hop_length,
            window="hann",
            freq_scale="no",
            # Do NOT define fmin/fmax here — freq_scale="no" + MelScale handles freq mapping
            center=center,
            pad_mode=pad_mode,
            iSTFT=False,
            sr=sample_rate,
            trainable=trainable,
            output_format="Complex",
            verbose=False,
        )
        self.normalized = normalized
        self.power = power
        self.register_buffer(
            "window",
            torch.hann_window(win_length if win_length is not None else n_fft),
        )
        self.trainable = trainable
        self.mel_scale = MelScale(
            n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, x: Tensor) -> Tensor:
        """waveform (B, T_audio) → mel power spectrogram (B, n_mels, T_frame)."""
        spec_f = self.spectrogram(x)  # (B, F_bins, T_frame, 2)  — last dim is (real, imag)
        if self.normalized:
            spec_f = spec_f / self.window.pow(2.0).sum().sqrt()
        if self.power is not None:
            # |spec|^power. Add small eps for grad safety when trainable.
            eps = 1e-8 if self.trainable else 0.0
            spec_f = torch.sqrt(spec_f[:, :, :, 0].pow(2) + spec_f[:, :, :, 1].pow(2) + eps)
            if self.power != 1.0:
                spec_f = spec_f.pow(self.power)
        mel_spec = self.mel_scale(spec_f)
        return mel_spec


def build_sydorskyi_mel_extractor(
    mel_params: dict | None = None,
    top_db: float = 80.0,
    amin: float = 1e-10,
    exportable: bool = True,
) -> nn.Sequential:
    """Build the full Sydorskyi mel chain as one nn.Sequential.

    Default `mel_params` matches selected_ebs.py / selected_eca.py:
        sample_rate=32000, n_mels=128, n_fft=2048, hop_length=512, f_min=20, normalized=True

    Chain:
        TraceableMelspec → ChannelAgnosticAmplitudeToDB(top_db=80, amin=1e-10) → NormalizeMelSpec
    """
    if mel_params is None:
        mel_params = {
            "sample_rate": 32000,
            "n_mels": 128,
            "f_min": 20,
            "n_fft": 2048,
            "hop_length": 512,
            "normalized": True,
        }
    return nn.Sequential(
        TraceableMelspec(**mel_params),
        ChannelAgnosticAmplitudeToDB(top_db=top_db, amin=amin),
        NormalizeMelSpec(exportable=exportable),
    )
