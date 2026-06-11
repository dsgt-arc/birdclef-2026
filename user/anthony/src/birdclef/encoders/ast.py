"""AST (Audio Spectrogram Transformer) encoder wrapper.

Uses HuggingFace's pre-trained AST model from MIT.
Reference: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
"""

import torch
import torchaudio

from .base import AudioEncoder


class ASTEncoder(AudioEncoder):
    """AST encoder for audio feature extraction.

    Uses the HuggingFace transformers implementation of AST.
    The model was pre-trained on ImageNet and fine-tuned on AudioSet.

    Args:
        model_name: HuggingFace model ID
        device: Device to run model on
    """

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: str = "cuda",
        pooling: str = "mean_all",
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        # Token-pooling strategy for the window embedding. AST prepends two
        # special tokens ([CLS] at 0, [DIST] at 1) before the patch grid.
        #   mean_all   — mean over ALL tokens incl. CLS+DIST (legacy default;
        #                kept for backward compat with existing callers)
        #   mean_patch — mean over patch tokens only ([:, 2:]); the convention
        #                for a fair cross-encoder probe (HEAR-style)
        #   cls        — mean of the CLS+DIST special tokens ([:, :2])
        if pooling not in ("mean_all", "mean_patch", "cls"):
            raise ValueError(f"Unknown pooling {pooling!r}")
        self.pooling = pooling

        from transformers import ASTModel, ASTFeatureExtractor

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        # AST-base has 768 hidden dim
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
            Features of shape (batch, embed_dim) - mean pooled over time
        """
        # Remove channel dimension: (batch, 1, samples) -> (batch, samples)
        waveform = waveform.squeeze(1)

        batch_size = waveform.shape[0]
        features_list = []

        for i in range(batch_size):
            # Convert to numpy for feature extractor (expects 16kHz)
            wav = waveform[i].cpu().numpy()

            # Use AST's feature extractor to compute mel spectrogram
            inputs = self.feature_extractor(
                wav,
                sampling_rate=16000,
                return_tensors="pt",
            )

            # Move to device
            input_values = inputs["input_values"].to(self.device)

            # Extract features
            outputs = self.model(input_values)

            # Pool the token sequence to one vector per the configured strategy.
            # Shape: (1, seq_len, hidden_size) -> (1, hidden_size).
            # Layout: [CLS, DIST, patch_0 ... patch_n].
            hidden_states = outputs.last_hidden_state
            if self.pooling == "cls":
                pooled = hidden_states[:, :2].mean(dim=1)
            elif self.pooling == "mean_patch":
                pooled = hidden_states[:, 2:].mean(dim=1)
            else:  # mean_all
                pooled = hidden_states.mean(dim=1)
            features_list.append(pooled)

        # Stack all features
        features = torch.cat(features_list, dim=0)  # (batch, embed_dim)

        return features
