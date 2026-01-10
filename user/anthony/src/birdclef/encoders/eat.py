"""EAT (Efficient Audio Transformer) encoder wrapper."""

import torch
import torchaudio

from .base import AudioEncoder


class EATEncoder(AudioEncoder):
    """EAT encoder for audio feature extraction.

    Uses EAT model via HuggingFace with trust_remote_code.

    Args:
        model_name: HuggingFace model ID (default: EAT-base pretrained)
        target_length: Target length in mel frames (512 for 5s, 1024 for 10s)
        device: Device to run model on
    """

    def __init__(
        self,
        model_name: str = "worstchan/EAT-base_epoch30_pretrain",
        target_length: int = 512,  # 5 seconds at 16kHz with 10ms hop
        device: str = "cuda",
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        super().__init__()
        self.device = device
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)

        # EAT-base has 768 hidden dim
        self._embed_dim = 768

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram.

        Args:
            waveform: (batch, samples) at 16kHz

        Returns:
            Mel spectrogram of shape (batch, 1, target_length, 128)
        """
        batch_size = waveform.shape[0]
        mels = []

        for i in range(batch_size):
            wav = waveform[i]
            # Normalize waveform
            wav = wav - wav.mean()

            # Compute mel-spectrogram using Kaldi fbank
            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )  # (time, 128)

            # Pad or truncate to target length
            n_frames = mel.shape[0]
            if n_frames < self.target_length:
                mel = torch.nn.functional.pad(
                    mel, (0, 0, 0, self.target_length - n_frames)
                )
            elif n_frames > self.target_length:
                mel = mel[: self.target_length, :]

            mels.append(mel)

        mel = torch.stack(mels, dim=0)  # (batch, time, 128)

        # Normalize
        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        # Add channel dimension: (batch, 1, time, 128)
        mel = mel.unsqueeze(1)

        return mel

    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio tensor of shape (batch, 1, num_samples) at 16kHz

        Returns:
            Features of shape (batch, embed_dim) - mean pooled over time
        """
        # Remove channel dimension: (batch, 1, samples) -> (batch, samples)
        waveform = waveform.squeeze(1)

        # Convert to mel-spectrogram
        mel = self._waveform_to_mel(waveform).to(self.device)

        # Extract features - returns (batch, seq_len+1, embed_dim) with CLS token
        features = self.model.extract_features(mel)

        # Mean pooling over time (excluding CLS token at position 0)
        features = features[:, 1:].mean(dim=1)  # (batch, embed_dim)

        return features
