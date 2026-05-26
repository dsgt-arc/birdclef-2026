"""Tucker-faithful mel spectrogram transform for exp 032.

Verbatim port of `MelSpecTransform` from
`vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.py`
(lines 400-407 + training loop lines 945-950).

This is a deliberate replacement for `NNAudioMelSpectrogram` on the exp 032
training path: the diagnostic at
`user/murilo/experiments/032-tucker-distilled-sed/scripts/diag_mel_pipeline.py`
showed that `NNAudioMelSpectrogram(norm_type="amplitude_to_db")` does THREE
normalizations (power_to_db → z-score → min-max [0,1]) where Tucker does
ZERO — Tucker's pipeline is just `power → AmplitudeToDB(80)`, with the
single per-sample z-score happening in the LightningModule's
`per_sample_mel_norm` step on top.

Combining `TuckerMelSpectrogram` with `DistilledSEDFinetuneModule(
per_sample_mel_norm=True)` reproduces Tucker's pipeline exactly:

    power_mel → AmplitudeToDB(top_db=80) → per-sample (mel - mean) / (std + eps)

The diagnostic confirmed this matches the kernel-side `compute_mel_tucker`
in `submission.py` to ~1e-7 element-wise. Training the model with this
transform should restore the training-inference distribution match that
collapsed exp 032's Phase 1 LB.

Reference:
- Tucker notebook lines 400-407 (`MelSpecTransform` class)
- Tucker notebook lines 945-950 (per-sample norm in training loop)
- Phase D § D.2 diagnostic at exp 032 scripts/diag_mel_pipeline.py
- Audit § 7.1 in analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class TuckerMelSpectrogram(nn.Module):
    """torchaudio MelSpectrogram + AmplitudeToDB(top_db=80). No normalization.

    Returns a 4D tensor (B, 1, n_mels, T) where the channel axis is inserted
    to match the rest of the codebase's mel-input convention (encoders expect
    (B, C, F, T)). The module's `per_sample_mel_norm` does the single
    per-sample z-score on top of this output.

    Args:
        sample_rate: Audio sample rate (Tucker: 32000).
        n_fft: FFT window size (Tucker: 2048).
        hop_length: Hop length in samples (Tucker: 512).
        n_mels: Number of mel filterbanks (Tucker: 256).
        fmin: Minimum frequency (Tucker: 20).
        fmax: Maximum frequency (Tucker: 16000).
        top_db: AmplitudeToDB dynamic range clamp (Tucker: 80).
        power: MelSpectrogram power (Tucker: 2.0).

    Notes on torchaudio defaults that Tucker INHERITS:
        - `norm=None` (no Slaney normalization on the filterbank)
        - `mel_scale="htk"` (htk mel filterbank construction)
        - `center=True` (reflective padding, gives `(N // hop_length) + 1` frames)
        - `power=2.0` (power spectrogram, NOT magnitude)

    For 5s @ 32 kHz + hop=512: output T = 160000 // 512 + 1 = 313 frames.

    Tucker uses `top_db=80` on AmplitudeToDB, which clamps the dB output to
    `db_multiplier * log10(amin)` floor; the relative magnitude floor is
    `mel_max - top_db` per spectrogram.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 256,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        top_db: float = 80.0,
        power: float = 2.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db
        self.power = power

        # Verbatim from Tucker's MelSpecTransform (lines 400-407)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=power,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram in dB scale (no normalization).

        Args:
            waveform: Input audio. Accepts (B, samples) or (B, 1, samples) or
                (samples,). For exp 032 training, the input is (B, 1, samples)
                from FocalDS / ScDS.

        Returns:
            Mel spectrogram in dB scale, shape (B, 1, n_mels, T). The
            module's `per_sample_mel_norm=True` step normalizes this
            downstream.
        """
        # Accept either (B, samples) or (B, 1, samples) or (samples,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 3:
            # (B, 1, samples) → (B, samples) for torchaudio MelSpec
            waveform = waveform.squeeze(1)
        # waveform now (B, samples)

        mel = self.mel_spec(waveform)  # (B, n_mels, T)
        mel = self.db_transform(mel)  # (B, n_mels, T)
        return mel.unsqueeze(1)  # (B, 1, n_mels, T) — match codebase convention

    def extra_repr(self) -> str:
        return (
            f"sr={self.sample_rate}, n_fft={self.n_fft}, hop={self.hop_length}, "
            f"n_mels={self.n_mels}, fmin={self.fmin}, fmax={self.fmax}, "
            f"top_db={self.top_db}, power={self.power}"
        )
