"""SSAMBA + classification head with masked mean pooling for variable-length audio."""

import torch
import torch.nn as nn
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.model import SSAMBAModel


class SSAMBAClassifier(nn.Module):
    """SSAMBA backbone with masked mean pooling and a linear classification head.

    Supports variable-length audio by accepting a padding mask that excludes
    padded tokens from the mean pooling operation.

    Args:
        config: SSAMBAConfig for the backbone
        num_classes: Number of output classes
        head_dropout: Dropout probability before the classification head
    """

    def __init__(self, config: SSAMBAConfig, num_classes: int, head_dropout: float = 0.3):
        super().__init__()
        self.backbone = SSAMBAModel(config)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(config.embed_dim, num_classes),
        )

    def forward(self, mel_specs, padding_mask=None):
        """Forward pass with optional masked mean pooling.

        Args:
            mel_specs: (B, 1, 128, T) mel-spectrograms
            padding_mask: (B, num_tokens) boolean tensor, True for real tokens.
                If None, uses backbone's built-in mean pooling.

        Returns:
            (B, num_classes) raw logits
        """
        if padding_mask is None:
            embeddings = self.backbone.forward_features(mel_specs)  # (B, embed_dim)
        else:
            # Get all token embeddings including CLS
            all_tokens = self.backbone.forward_features(
                mel_specs, return_all_tokens=True
            )  # (B, num_tokens + 1, embed_dim)

            # Remove CLS token (inserted at middle position)
            cls_pos = self.backbone._cls_token_position
            patch_tokens = torch.cat(
                [all_tokens[:, :cls_pos, :], all_tokens[:, cls_pos + 1 :, :]],
                dim=1,
            )  # (B, num_tokens, embed_dim)

            # Masked mean pooling: only aggregate real tokens
            mask = padding_mask.unsqueeze(-1).to(patch_tokens.dtype)  # (B, num_tokens, 1)
            masked_sum = (patch_tokens * mask).sum(dim=1)  # (B, embed_dim)
            mask_count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
            embeddings = masked_sum / mask_count

        return self.head(embeddings)
