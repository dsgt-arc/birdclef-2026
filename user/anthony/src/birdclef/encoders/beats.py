"""BEATs encoder wrapper."""

import sys
from pathlib import Path

import torch

from .base import AudioEncoder

# Add vendor path for BEATs imports
VENDOR_PATH = Path(__file__).parents[5] / "vendor" / "unilm" / "beats"


class BEATsEncoder(AudioEncoder):
    """BEATs encoder for audio feature extraction.

    Uses the BEATs model pretrained on AudioSet.

    Args:
        checkpoint_path: Path to BEATs checkpoint (.pt file)
        device: Device to run model on
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Import BEATs from vendor directory
        if str(VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(VENDOR_PATH))

        from BEATs import BEATs, BEATsConfig

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(device)

        self._embed_dim = cfg.encoder_embed_dim

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

        # BEATs extract_features returns (features, padding_mask)
        features, _ = self.model.extract_features(waveform)
        # features shape: (batch, seq_len, embed_dim)

        # Mean pooling over time
        features = features.mean(dim=1)  # (batch, embed_dim)

        return features
