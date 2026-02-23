"""BEATs encoder wrapper.

Adapted from user/anthony/src/birdclef/encoders/beats.py for fine-tuning
(gradients enabled, no hardcoded device management).

Requires:
  1. Initialize the vendor/unilm submodule: git submodule update --init vendor/unilm
  2. Download checkpoint: BEATs_iter3_plus_AS2M.pt
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Vendor path for BEATs imports (relative to repo root)
VENDOR_PATH = Path(__file__).parents[5] / "vendor" / "unilm" / "beats"


class BEATsEncoder(nn.Module):
    """BEATs encoder for audio feature extraction.

    Loads from a local checkpoint file. Uses vendor code from vendor/unilm/beats/.

    Args:
        checkpoint_path: Path to BEATs checkpoint (.pt file)
    """

    def __init__(self, checkpoint_path: str | Path):
        super().__init__()

        # Import BEATs from vendor directory
        if str(VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(VENDOR_PATH))

        from BEATs import BEATs, BEATsConfig

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])

        self._embed_dim = cfg.encoder_embed_dim

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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio tensor of shape (batch, samples) at 16kHz

        Returns:
            Features of shape (batch, embed_dim) — mean pooled over time
        """
        # BEATs extract_features returns (features, padding_mask)
        features, _ = self.model.extract_features(waveform)
        # features shape: (batch, seq_len, embed_dim)

        # Mean pooling over time
        features = features.mean(dim=1)  # (batch, embed_dim)

        return features
