"""EAT (Efficient Audio Transformer) encoder wrapper.

Adapted from user/anthony/src/birdclef/encoders/eat.py for fine-tuning
(gradients enabled, no hardcoded device management).
"""

import torch
import torch.nn as nn
import torchaudio


class EATEncoder(nn.Module):
    """EAT encoder for audio feature extraction.

    Loads the EAT-base model from HuggingFace, converts waveforms to Kaldi fbank
    features, and extracts embeddings via mean pooling (excluding CLS token).

    Args:
        model_name: HuggingFace model ID
        target_length: Target length in mel frames (512 for 5s at 16kHz with 10ms hop)
        norm_mean: Normalization mean for fbank features
        norm_std: Normalization std for fbank features
    """

    def __init__(
        self,
        model_name: str = "worstchan/EAT-base_epoch30_pretrain",
        target_length: int = 512,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        super().__init__()
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self._embed_dim = 768

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def freeze(self):
        """Freeze all encoder parameters (for linear probing)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        """Unfreeze all encoder parameters (for fine-tuning)."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized Kaldi fbank features.

        Args:
            waveform: (batch, samples) at 16kHz

        Returns:
            Fbank features of shape (batch, 1, target_length, 128)
        """
        batch_size = waveform.shape[0]
        mels = []

        for i in range(batch_size):
            wav = waveform[i]
            # Normalize waveform
            wav = wav - wav.mean()

            # Compute fbank using Kaldi conventions
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
                mel = torch.nn.functional.pad(mel, (0, 0, 0, self.target_length - n_frames))
            elif n_frames > self.target_length:
                mel = mel[: self.target_length, :]

            mels.append(mel)

        mel = torch.stack(mels, dim=0)  # (batch, time, 128)

        # Normalize
        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        # Add channel dimension: (batch, 1, time, 128)
        mel = mel.unsqueeze(1)

        return mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio tensor of shape (batch, samples) at 16kHz

        Returns:
            Features of shape (batch, 768) — mean pooled over time
        """
        # Convert to fbank
        fbank = self._waveform_to_fbank(waveform)

        # Extract features — returns (batch, seq_len+1, embed_dim) with CLS token
        features = self.model.extract_features(fbank)

        # Mean pooling over time (excluding CLS token at position 0)
        features = features[:, 1:].mean(dim=1)  # (batch, 768)

        return features
