"""EfficientNet-B1 frozen backbone with SED head (frame + attention pooling).

Architecture:
    waveform → nnAudio mel → SpecAugment (train)
    → frozen EfficientNet-B1 backbone → (B, 1280, 8, 20)
    → mean pool freq dim → (B, 1280, 20)
    → avg pool time 4:1 → (B, 1280, 5)
    → transpose → (B, 5, 1280)
    ┌→ frame_head: Dropout + Linear → frame_logits (B, 5, C)
    └→ att_head: Linear → tanh → softmax → att_weights (B, 5, C)
    → clip_pred = sum(att_weights * sigmoid(frame_logits)) → (B, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EfficientNetModel


class EfficientNetSED(nn.Module):
    """Frozen EfficientNet-B1 backbone with SED frame head + attention pooling.

    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of output classes
        mel_spec_layer: nnAudio MelSpectrogram nn.Module instance
        target_frames: Expected number of mel frames
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        head_dropout: Dropout probability before frame head
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
        embed_dim = self.backbone.config.hidden_dim  # 1280

        # Pool 20 time frames → 5 (matching 1s soft label resolution)
        self.time_pool = nn.AvgPool1d(kernel_size=4)

        # Per-frame classification head
        self.frame_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Attention head for clip-level aggregation
        self.att_head = nn.Linear(embed_dim, num_classes)

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
        """Forward pass from raw waveforms to frame logits + clip prediction.

        Args:
            waveforms: (B, num_samples) raw audio
            padding_mask: unused, kept for interface compat
            spec_augment: Optional SpecAugment callable (training only)

        Returns:
            clip_pred: (B, C) clip-level probabilities from attention pooling
            frame_logits: (B, 5, C) per-frame raw logits
        """
        mel = self.compute_mel(waveforms)  # (B, 1, n_mels, T)

        if spec_augment is not None:
            mel = spec_augment(mel)

        # Forward through frozen backbone
        with torch.no_grad():
            outputs = self.backbone(pixel_values=mel)
            features = outputs.last_hidden_state  # (B, 1280, H=8, W=20)

        # Mean pool frequency dimension: (B, 1280, 8, 20) → (B, 1280, 20)
        features = features.mean(dim=2)

        # Pool time 20 → 5: (B, 1280, 20) → (B, 1280, 5)
        features = self.time_pool(features)

        # Transpose to (B, 5, 1280) for per-frame heads
        features = features.transpose(1, 2)

        # Per-frame logits
        frame_logits = self.frame_head(features)  # (B, 5, C)

        # Attention-weighted clip prediction
        att_scores = self.att_head(features)  # (B, 5, C)
        att_weights = torch.softmax(torch.tanh(att_scores), dim=1)  # (B, 5, C)
        clip_pred = (att_weights * torch.sigmoid(frame_logits)).sum(dim=1)  # (B, C)

        return clip_pred, frame_logits
