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
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

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

            # Get last hidden state and mean pool over sequence
            # Shape: (1, seq_len, hidden_size) -> (1, hidden_size)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)
            features_list.append(pooled)

        # Stack all features
        features = torch.cat(features_list, dim=0)  # (batch, embed_dim)

        return features
