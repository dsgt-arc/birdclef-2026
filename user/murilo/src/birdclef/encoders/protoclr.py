"""ProtoCLR (CvT-13) encoder wrapper for bioacoustic feature extraction.

Loads the ProtoCLR model from HuggingFace Hub (ilyassmoummad/ProtoCLR).
The CvT architecture is defined in the HF repo (not in transformers or timm),
so we download and dynamically import it.

Reference: https://github.com/ilyassmoummad/ProtoCLR
Model card: https://huggingface.co/ilyassmoummad/ProtoCLR
"""

import importlib.util
import json
import logging
import os
import sys

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _load_cvt_class(model_dir: str):
    """Import ConvolutionalVisionTransformer from the downloaded HF repo files."""
    cvt_path = os.path.join(model_dir, "cvt.py")

    # Add model_dir to sys.path so cvt.py's internal imports resolve
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location("protoclr_cvt", cvt_path)
    cvt_module = importlib.util.module_from_spec(spec)
    sys.modules["protoclr_cvt"] = cvt_module
    spec.loader.exec_module(cvt_module)

    return cvt_module.ConvolutionalVisionTransformer


class ProtoCLREncoder(nn.Module):
    """CvT-13 encoder pretrained with ProtoCLR on Xeno-Canto.

    Downloads model architecture and weights from HuggingFace Hub.
    Returns (batch, 384) embeddings via mean pooling or CLS token.

    Args:
        model_name: HuggingFace repo ID
        checkpoint: Weight file name in the repo
    """

    def __init__(
        self,
        model_name: str = "ilyassmoummad/ProtoCLR",
        checkpoint: str = "protoclr.pth",
    ):
        super().__init__()

        from huggingface_hub import snapshot_download

        # Download model files from HuggingFace Hub
        model_dir = snapshot_download(
            model_name,
            allow_patterns=["*.py", "*.json", checkpoint],
        )
        logger.info(f"ProtoCLR model files downloaded to {model_dir}")

        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Import and instantiate CvT model
        CvT = _load_cvt_class(model_dir)
        self.model = CvT(
            in_chans=1,
            num_classes=1000,  # Match checkpoint; head will be unused
            spec=config["MODEL"]["SPEC"],
        )

        # Load pretrained weights
        weights_path = os.path.join(model_dir, checkpoint)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded ProtoCLR weights from {checkpoint}")

        # Replace classification head (we use our own in LoRAFinetuneModule)
        self.model.head = nn.Identity()

        self._embed_dim = config["MODEL"]["SPEC"]["DIM_EMBED"][-1]  # 384

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: mel spectrogram -> feature vector.

        Args:
            x: Input tensor, shape (batch, 1, n_mels, time)

        Returns:
            Feature vector, shape (batch, embed_dim)
        """
        return self.model.forward_features(x)
