"""EfficientNet-B1 frozen backbone with linear classification head.

Architecture:
    waveform → nnAudio mel → SpecAugment (train)
    → frozen EfficientNet-B1 backbone → adaptive avg pool → 1280-dim
    → Dropout → Linear → clip logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features import MelSpectrogram
from transformers import EfficientNetModel


class EfficientNetClassifier(nn.Module):
    """Frozen EfficientNet-B1 backbone with trainable linear head.

    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of output classes
        mel_spec_layer: nnAudio MelSpectrogram nn.Module instance
        target_frames: Expected number of mel frames
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        head_dropout: Dropout probability before classification head
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        mel_spec_layer: nn.Module,
        target_frames: int = 625,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.mel_spec = mel_spec_layer
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Load pretrained backbone and freeze all params
        self.backbone = EfficientNetModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # EfficientNet-B1 output: 1280 channels
        embed_dim = self.backbone.config.hidden_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def train(self, mode=True):
        """Override to keep backbone in eval mode (frozen BN stats)."""
        super().train(mode)
        self.backbone.eval()
        return self

    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute normalized mel spectrogram from raw waveforms.

        Args:
            waveforms: (B, num_samples) raw audio

        Returns:
            (B, 1, n_mels, target_frames) normalized mel spectrogram
        """
        mel = self.mel_spec(waveforms)  # (B, n_mels, T)
        mel = torch.log(mel.clamp(min=1e-10))
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        T_mel = mel.shape[2]
        if T_mel > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T_mel < self.target_frames:
            pad_len = self.target_frames - T_mel
            mel = F.pad(mel, (0, pad_len), value=0)
        return mel.unsqueeze(1)  # (B, 1, n_mels, T)

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spec_augment=None,
    ):
        """Forward pass from raw waveforms to clip-level logits.

        Args:
            waveforms: (B, num_samples) raw audio
            padding_mask: (B, T) boolean — unused, kept for interface compat
            spec_augment: Optional SpecAugment callable (training only)

        Returns:
            clip_logits: (B, C) clip-level raw logits
        """
        mel = self.compute_mel(waveforms)  # (B, 1, n_mels, T)

        if spec_augment is not None:
            mel = spec_augment(mel)

        # Forward through frozen backbone (BirdSet model expects 1-channel mel)
        with torch.no_grad():
            outputs = self.backbone(pixel_values=mel)
            # last_hidden_state: (B, C, H, W) — use pooler_output or manual pool
            features = outputs.last_hidden_state  # (B, 1280, H', W')
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, 1280)

        clip_logits = self.head(pooled)  # (B, C)
        return clip_logits
