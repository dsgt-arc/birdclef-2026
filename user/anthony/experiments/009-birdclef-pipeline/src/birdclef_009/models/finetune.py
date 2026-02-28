"""SSAMBA clip-level classifier with mean pooling.

Architecture:
    waveform → nnAudio mel → SpecAugment (train) → SSAMBA backbone
    → mean pool patch tokens → classification head → clip logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.model import SSAMBAModel


class SSAMBAClassifier(nn.Module):
    """SSAMBA backbone with mean-pooled clip-level classification.

    For inference, use forward_tokens() to get per-token embeddings,
    then pool over arbitrary windows and apply self.head.

    Args:
        config: SSAMBAConfig for the backbone
        num_classes: Number of output classes
        mel_spec_layer: nnAudio MelSpectrogram nn.Module instance
        target_frames: Expected number of mel frames
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        head_dropout: Dropout probability before classification head
    """

    def __init__(
        self,
        config: SSAMBAConfig,
        num_classes: int,
        mel_spec_layer: nn.Module,
        target_frames: int = 1000,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.mel_spec = mel_spec_layer
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.backbone = SSAMBAModel(config)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(config.embed_dim, num_classes),
        )

    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute normalized mel spectrogram from raw waveforms.

        Args:
            waveforms: (B, num_samples) raw audio

        Returns:
            (B, 1, n_mels, target_frames) normalized mel spectrogram
        """
        mel = self.mel_spec(waveforms)
        mel = torch.log(mel.clamp(min=1e-10))
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        T_mel = mel.shape[2]
        if T_mel > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T_mel < self.target_frames:
            pad_len = self.target_frames - T_mel
            mel = F.pad(mel, (0, pad_len), value=0)
        return mel.unsqueeze(1)

    def _extract_patch_tokens(self, mel: torch.Tensor) -> torch.Tensor:
        """Run backbone and extract patch tokens (without CLS).

        Args:
            mel: (B, 1, n_mels, T) mel spectrogram

        Returns:
            (B, T, D) patch token embeddings
        """
        all_tokens = self.backbone.forward_features(mel, return_all_tokens=True)
        # Remove CLS token
        cls_pos = self.backbone._cls_token_position
        patch_tokens = torch.cat(
            [all_tokens[:, :cls_pos, :], all_tokens[:, cls_pos + 1:, :]],
            dim=1,
        )
        return patch_tokens

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spec_augment=None,
    ):
        """Forward pass from raw waveforms to clip-level logits.

        Args:
            waveforms: (B, num_samples) raw audio
            padding_mask: (B, T) boolean, True for real tokens
            spec_augment: Optional SpecAugment callable (training only)

        Returns:
            clip_logits: (B, C) clip-level raw logits
        """
        mel = self.compute_mel(waveforms)

        if spec_augment is not None:
            mel = spec_augment(mel)

        patch_tokens = self._extract_patch_tokens(mel)  # (B, T, D)

        # Mean pool with padding mask
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (patch_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = patch_tokens.mean(dim=1)  # (B, D)

        clip_logits = self.head(pooled)  # (B, C)
        return clip_logits

    @torch.no_grad()
    def forward_windows(
        self,
        waveforms: torch.Tensor,
        window_size: int = 500,
    ) -> torch.Tensor:
        """Inference: get per-window predictions by pooling token windows.

        Splits tokens into non-overlapping windows of window_size, mean-pools
        each window, and applies the classification head.

        Args:
            waveforms: (B, num_samples) raw audio
            window_size: Number of tokens per window (500 = 5s at 128x1)

        Returns:
            (B, num_windows, C) per-window logits
        """
        mel = self.compute_mel(waveforms)
        patch_tokens = self._extract_patch_tokens(mel)  # (B, T, D)
        B, T, D = patch_tokens.shape

        num_windows = T // window_size
        windows = patch_tokens[:, :num_windows * window_size, :].reshape(
            B, num_windows, window_size, D
        )
        pooled = windows.mean(dim=2)  # (B, num_windows, D)
        logits = self.head(pooled)  # (B, num_windows, C)
        return logits
