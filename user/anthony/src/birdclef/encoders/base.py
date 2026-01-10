"""Base audio encoder interface."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class AudioEncoder(ABC, nn.Module):
    """Base class for audio encoders."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio tensor of shape (batch, 1, num_samples) at 16kHz

        Returns:
            Features of shape (batch, embed_dim) - pooled utterance-level features
        """
        pass

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass - alias for extract_features."""
        return self.extract_features(waveform)
