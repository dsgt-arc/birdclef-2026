"""EfficientNet-B1 with optional LoRA adapters and configurable pooling.

Supports three backbone modes:
  - frozen: backbone frozen, only head trains (exp 012 baseline)
  - lora: backbone frozen + LoRA adapters
  - full: backbone fully unfrozen

Supports two pooling modes:
  - avg: global average pooling (simple, fast)
  - attention: attention-weighted pooling (train) / framewise max (inference)

Architecture:
    waveform → nnAudio mel → SpecAugment (train)
    → EfficientNet-B1 backbone (+ optional LoRA)
    → mean over freq → (B, 1280, T) frame features
    → pooling → (B, C) clip logits
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EfficientNetModel

logger = logging.getLogger(__name__)


class AttentionPoolingHead(nn.Module):
    """SED-style attention pooling + frame classification head.

    Training: attention-weighted mean of frame logits
    Inference: framewise max of frame logits

    This is the dominant pattern from BirdCLEF 2025 top solutions.

    Args:
        embed_dim: Input feature dimension (1280 for EfficientNet-B1)
        num_classes: Number of output classes
        dropout: Dropout before linear layers
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.frame_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.att_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (B, T, D) per-frame features

        Returns:
            clip_logits: (B, C) clip-level logits
        """
        frame_logits = self.frame_head(frame_features)  # (B, T, C)
        att_logits = self.att_head(frame_features)       # (B, T, C)

        if self.training:
            # Attention-weighted mean: softmax over time, weighted sum of frame probs
            att_weights = torch.softmax(att_logits, dim=1)  # (B, T, C)
            clip_logits = (att_weights * frame_logits).sum(dim=1)  # (B, C)
        else:
            # Framewise max: take the highest frame prediction per species
            clip_logits = frame_logits.max(dim=1).values  # (B, C)

        return clip_logits


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B1 backbone with configurable fine-tuning and pooling.

    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of output classes
        mel_spec_layer: nnAudio MelSpectrogram nn.Module instance
        target_frames: Expected number of mel frames
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        head_dropout: Dropout probability before classification head
        backbone_mode: "frozen" | "lora" | "full"
        pooling_mode: "avg" | "attention"
        lora_rank: LoRA rank (only used if backbone_mode="lora")
        lora_alpha: LoRA alpha scaling (only used if backbone_mode="lora")
        lora_dropout: LoRA dropout (only used if backbone_mode="lora")
        lora_targets: List of module name patterns to apply LoRA to
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
        backbone_mode: str = "frozen",
        pooling_mode: str = "avg",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_targets: list[str] | None = None,
    ):
        super().__init__()
        self.mel_spec = mel_spec_layer
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.backbone_mode = backbone_mode
        self.pooling_mode = pooling_mode

        # Load pretrained backbone
        self.backbone = EfficientNetModel.from_pretrained(model_name)
        embed_dim = self.backbone.config.hidden_dim

        if backbone_mode == "frozen":
            self._freeze_backbone()
        elif backbone_mode == "lora":
            self._apply_lora(lora_rank, lora_alpha, lora_dropout, lora_targets)
        elif backbone_mode == "full":
            logger.info("Backbone fully unfrozen — all params trainable")
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        # Classification head
        if pooling_mode == "avg":
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(embed_dim, num_classes),
            )
        elif pooling_mode == "attention":
            self.head = AttentionPoolingHead(embed_dim, num_classes, dropout=head_dropout)
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

        logger.info(f"Pooling mode: {pooling_mode}")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info("Backbone frozen — only head trains")

    def _apply_lora(self, rank, alpha, dropout, targets):
        from peft import LoraConfig, get_peft_model

        if targets is None:
            targets = ["project_conv", "expand_conv", "reduce", "expand"]

        for param in self.backbone.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets, bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"LoRA applied — {trainable:,} trainable / {total:,} total backbone params "
            f"({100*trainable/total:.1f}%)"
        )

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_mode in ("frozen", "lora"):
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        return self

    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(waveforms)
        mel = torch.log(mel.clamp(min=1e-10))
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        T_mel = mel.shape[2]
        if T_mel > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T_mel < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - T_mel), value=0)
        return mel.unsqueeze(1)

    def _extract_features(self, mel: torch.Tensor) -> torch.Tensor:
        """Run backbone and return features.

        Args:
            mel: (B, 1, n_mels, T) mel spectrogram

        Returns:
            features: (B, 1280, H, W) backbone feature map
        """
        if self.backbone_mode == "frozen":
            with torch.no_grad():
                return self.backbone(pixel_values=mel).last_hidden_state
        else:
            return self.backbone(pixel_values=mel).last_hidden_state

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spec_augment=None,
    ):
        """Forward pass from raw waveforms to clip-level logits.

        Args:
            waveforms: (B, num_samples) raw audio
            padding_mask: unused, kept for interface compat
            spec_augment: Optional SpecAugment callable (training only)

        Returns:
            clip_logits: (B, C) clip-level raw logits
        """
        mel = self.compute_mel(waveforms)

        if spec_augment is not None:
            mel = spec_augment(mel)

        features = self._extract_features(mel)  # (B, 1280, H, W)

        if self.pooling_mode == "avg":
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, 1280)
            clip_logits = self.head(pooled)
        elif self.pooling_mode == "attention":
            # Collapse freq, keep time: (B, 1280, H, W) → (B, W, 1280)
            frame_features = features.mean(dim=2).permute(0, 2, 1)  # (B, T, 1280)
            clip_logits = self.head(frame_features)

        return clip_logits
