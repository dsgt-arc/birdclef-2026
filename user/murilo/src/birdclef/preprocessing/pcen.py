"""Per-Channel Energy Normalization (PCEN) for mel spectrograms.

PCEN (Wang et al. 2017) applies adaptive gain control and dynamic range
compression, producing features invariant to recording conditions.

Formula: PCEN(E) = (E / (ε + M)^α + δ)^r - δ^r
  E: mel spectrogram energy (linear power)
  M: smoothed energy via IIR filter: M_t = (1-s) * M_{t-1} + s * E_t
  s: smoothing coefficient (time constant)
  α: AGC strength
  δ: bias (root compression offset)
  r: compression exponent
  ε: numerical stability constant

Reference: https://github.com/daemon/pytorch-pcen
"""

import torch
import torch.nn as nn


class PCENLayer(nn.Module):
    """Per-Channel Energy Normalization layer.

    Operates on (B, n_mels, T) tensors in linear power scale.
    Supports per-channel (per-mel-bin) learnable parameters.

    Args:
        n_mels: Number of mel bins (for per-channel params).
        s: Smoothing coefficient for IIR filter (default: 0.025).
        alpha: AGC strength (default: 0.98).
        delta: Bias term (default: 2.0).
        r: Compression exponent (default: 0.5).
        eps: Numerical stability (default: 1e-6).
        trainable: If True, s/alpha/delta/r are nn.Parameters.
        per_channel: If True, separate params per mel bin.
    """

    def __init__(
        self,
        n_mels: int = 256,
        s: float = 0.025,
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
        eps: float = 1e-6,
        trainable: bool = True,
        per_channel: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.trainable = trainable
        self.n_mels = n_mels

        size = (n_mels,) if per_channel else (1,)

        if trainable:
            # Raw (unconstrained) parameters — clamped in properties
            self.s_param = nn.Parameter(torch.full(size, s))
            self.alpha_param = nn.Parameter(torch.full(size, alpha))
            self.delta_param = nn.Parameter(torch.full(size, delta))
            self.r_param = nn.Parameter(torch.full(size, r))
        else:
            self.register_buffer("s_buf", torch.full(size, s))
            self.register_buffer("alpha_buf", torch.full(size, alpha))
            self.register_buffer("delta_buf", torch.full(size, delta))
            self.register_buffer("r_buf", torch.full(size, r))

    @property
    def s(self):
        if self.trainable:
            return self.s_param.clamp(1e-4, 1.0)
        return self.s_buf

    @property
    def alpha(self):
        if self.trainable:
            return self.alpha_param.clamp(1e-4, 1.0)
        return self.alpha_buf

    @property
    def delta(self):
        if self.trainable:
            return self.delta_param.clamp(1e-4)
        return self.delta_buf

    @property
    def r(self):
        if self.trainable:
            return self.r_param.clamp(1e-4, 1.0)
        return self.r_buf

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to mel spectrogram.

        Args:
            mel: (B, n_mels, T) linear power mel spectrogram.

        Returns:
            (B, n_mels, T) PCEN-normalized spectrogram.
        """
        s = self.s
        alpha = self.alpha
        delta = self.delta
        r = self.r

        # Broadcast shapes: (n_mels,) → (1, n_mels, 1)
        s_bc = s.unsqueeze(0).unsqueeze(-1)
        alpha_bc = alpha.unsqueeze(0).unsqueeze(-1)
        delta_bc = delta.unsqueeze(0).unsqueeze(-1)
        r_bc = r.unsqueeze(0).unsqueeze(-1)

        # IIR smoothing: M_t = (1 - s) * M_{t-1} + s * E_t
        T = mel.shape[-1]
        m_frames = [mel[..., :1]]  # M_0 = E_0

        for t in range(1, T):
            m_t = (1 - s_bc) * m_frames[-1] + s_bc * mel[..., t : t + 1]
            m_frames.append(m_t)

        M = torch.cat(m_frames, dim=-1)  # (B, n_mels, T)

        # PCEN: (E / (eps + M)^alpha + delta)^r - delta^r
        smooth = (self.eps + M).pow(alpha_bc)
        pcen_out = (mel / smooth + delta_bc).pow(r_bc) - delta_bc.pow(r_bc)

        return pcen_out

    def get_params_dict(self) -> dict:
        """Return current parameter values as a plain dict (for serialization)."""
        return {
            "s": self.s.detach().cpu().tolist(),
            "alpha": self.alpha.detach().cpu().tolist(),
            "delta": self.delta.detach().cpu().tolist(),
            "r": self.r.detach().cpu().tolist(),
            "eps": self.eps,
            "n_mels": self.n_mels,
            "trainable": self.trainable,
        }
