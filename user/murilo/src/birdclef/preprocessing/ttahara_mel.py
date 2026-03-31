"""Faithful reproduction of ttahara's LogMelSpectrogramTransform.

Source: kaggle kernels pull ttahara/birdclef-2026-hgnetv2-b0-baseline-training
Score: 0.884 LB (4-fold ensemble, OpenVINO CPU inference)

Pipeline:
    torchaudio.MelSpectrogram → AmplitudeToDB(power, top_db=80)
    → Resize(256, 256) → per-sample min-max [0, 1] → (B, 1, 256, 256)

Key differences from NNAudioMelSpectrogram:
    - Uses torchaudio (not nnAudio) for exact parameter parity
    - Explicit win_length (626) separate from n_fft (2048)
    - Resize to fixed square shape (256x256)
    - Per-sample min-max normalization only (no standardization)
"""

import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import v2 as tvt_v2


class TtaharaMelTransform(nn.Module):
    """Exact reproduction of ttahara's mel spectrogram transform.

    Args:
        sample_rate: Audio sample rate (default: 32000).
        n_fft: FFT size (default: 2048).
        win_length: STFT window length in samples (default: 626, ~20ms at 32kHz).
        hop_length: STFT hop length in samples (default: 313, ~10ms at 32kHz).
        f_min: Minimum frequency for mel filterbank (default: 20).
        n_mels: Number of mel bands (default: 256).
        power: Exponent for magnitude spectrogram (default: 2.0 = power spectrum).
        top_db: AmplitudeToDB top_db clipping (default: 80.0).
        target_shape: Resize target as (height, width) tuple (default: (256, 256)).
        out_channels: Number of output channels (default: 1). Set to 3 to repeat
            mono mel to 3 channels, preserving full RGB pretrained weights.
    """

    def __init__(
        self,
        sample_rate: int = 32_000,
        n_fft: int = 2048,
        win_length: int = 626,
        hop_length: int = 313,
        f_min: float = 20,
        n_mels: int = 256,
        power: float = 2.0,
        top_db: float = 80.0,
        target_shape: tuple[int, int] = (256, 256),
        out_channels: int = 1,
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            n_mels=n_mels,
            power=power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize = tvt_v2.Resize(size=target_shape)
        self.out_channels = out_channels

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to normalized log-mel spectrogram.

        Args:
            wave: (B, samples) or (samples,) raw waveform.

        Returns:
            (B, 1, H, W) or (1, H, W) normalized log-mel spectrogram.
        """
        single = wave.dim() == 1
        if single:
            wave = wave.unsqueeze(0)

        mel_spec = self.mel_transform(wave)  # (B, n_mels, time)
        lms = self.db(mel_spec)  # (B, n_mels, time)
        lms = self.resize(lms)  # (B, target_h, target_w)

        # Per-sample min-max normalization to [0, 1]
        batch_size = lms.shape[0]
        lms_flat = lms.reshape(batch_size, -1)
        lms_min = lms_flat.min(dim=1)[0][:, None, None]
        lms_max = lms_flat.max(dim=1)[0][:, None, None]
        lms = (lms - lms_min) / (lms_max - lms_min + 1e-7)

        result = lms[:, None, :, :]  # (B, 1, H, W)
        if self.out_channels > 1:
            result = result.repeat(1, self.out_channels, 1, 1)  # (B, C, H, W)
        if single:
            result = result.squeeze(0)
        return result
