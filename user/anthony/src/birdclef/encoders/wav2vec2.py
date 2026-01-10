"""Wav2Vec2 encoder wrapper."""

import torch
from transformers import Wav2Vec2Model

from .base import AudioEncoder


class Wav2Vec2Encoder(AudioEncoder):
    """Wav2Vec2 encoder for audio feature extraction.

    Uses the base wav2vec2 model pretrained on LibriSpeech.

    Args:
        model_name: HuggingFace model name (default: wav2vec2-base)
        device: Device to run model on
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self._embed_dim = self.model.config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio tensor of shape (batch, 1, num_samples) at 16kHz

        Returns:
            Features of shape (batch, embed_dim) - mean pooled
        """
        # Remove channel dimension: (batch, 1, samples) -> (batch, samples)
        waveform = waveform.squeeze(1).to(self.device)

        # Extract features
        outputs = self.model(waveform)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean pooling over time
        features = hidden_states.mean(dim=1)  # (batch, hidden_size)

        return features
