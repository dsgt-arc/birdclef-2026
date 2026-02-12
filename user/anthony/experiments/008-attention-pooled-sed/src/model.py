"""SSAMBA SED classifier with nnAudio mel, per-token SED head, and attention pooling.

Architecture:
    waveform → nnAudio mel → SpecAugment (train) → SSAMBA backbone → SED head
    → attention pooling → clip prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.model import SSAMBAModel


class SSAMBASEDClassifier(nn.Module):
    """SSAMBA backbone with per-token SED head and attention-pooled clip prediction.

    Integrates nnAudio MelSpectrogram as a differentiable front-end.

    Args:
        config: SSAMBAConfig for the backbone
        num_classes: Number of output classes
        mel_spec_layer: nnAudio MelSpectrogram nn.Module instance
        target_frames: Expected number of mel frames (for truncation/padding)
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        head_dropout: Dropout probability before SED head
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
        self.sed_head = nn.Sequential(
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
        # nnAudio returns (B, n_mels, T_mel) in linear power scale
        mel = self.mel_spec(waveforms)
        # Convert to log scale to match kaldi fbank (which returns log filterbank energies)
        mel = torch.log(mel.clamp(min=1e-10))
        # Normalize with AudioSet statistics
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        # Truncate or pad to target_frames
        T_mel = mel.shape[2]
        if T_mel > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T_mel < self.target_frames:
            pad_len = self.target_frames - T_mel
            mel = F.pad(mel, (0, pad_len), value=0)
        # Add channel dim: (B, n_mels, T) → (B, 1, n_mels, T)
        return mel.unsqueeze(1)

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spec_augment=None,
    ):
        """Forward pass from raw waveforms to clip + SED predictions.

        Args:
            waveforms: (B, num_samples) raw audio
            padding_mask: (B, T) boolean, True for real tokens
            spec_augment: Optional SpecAugment callable (training only)

        Returns:
            clip_pred: (B, C) clip-level probabilities from attention pooling
            sed_logits: (B, T, C) per-token raw logits
        """
        # Compute mel spectrogram on GPU
        mel = self.compute_mel(waveforms)

        # Apply SpecAugment during training
        if spec_augment is not None:
            mel = spec_augment(mel)

        # Get all token embeddings from backbone
        all_tokens = self.backbone.forward_features(mel, return_all_tokens=True)
        # all_tokens: (B, num_patches + 1, D) — includes CLS token

        # Remove CLS token (middle position)
        cls_pos = self.backbone._cls_token_position
        patch_tokens = torch.cat(
            [all_tokens[:, :cls_pos, :], all_tokens[:, cls_pos + 1:, :]],
            dim=1,
        )  # (B, T, D)

        # Per-token SED logits
        sed_logits = self.sed_head(patch_tokens)  # (B, T, C)

        # Attention pooling: softmax over tokens, weighted sum of sigmoid probs
        if padding_mask is not None:
            # Mask padded tokens with -inf before softmax
            sed_for_att = sed_logits.masked_fill(~padding_mask.unsqueeze(-1), float("-inf"))
        else:
            sed_for_att = sed_logits

        att_weights = F.softmax(sed_for_att, dim=1)  # (B, T, C)
        att_weights = att_weights.nan_to_num(0.0)     # safety: all-padded → NaN → 0
        token_probs = torch.sigmoid(sed_logits)       # (B, T, C)
        clip_pred = (att_weights * token_probs).sum(dim=1)  # (B, C)

        return clip_pred, sed_logits
