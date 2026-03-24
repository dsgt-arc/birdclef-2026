"""EfficientNet-B1 with optional LoRA adapters.

Forked from exp 012. Supports three modes:
  - frozen: backbone frozen, only head trains (exp 012 baseline)
  - lora: backbone frozen + LoRA adapters on attention layers
  - full: backbone fully unfrozen (not recommended, use LoRA instead)

Architecture:
    waveform → nnAudio mel → SpecAugment (train)
    → EfficientNet-B1 backbone (+ optional LoRA) → adaptive avg pool → 1280-dim
    → Dropout → Linear → clip logits
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EfficientNetModel

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B1 backbone with configurable fine-tuning strategy.

    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of output classes
        mel_spec_layer: nnAudio MelSpectrogram nn.Module instance
        target_frames: Expected number of mel frames
        norm_mean: Mel normalization mean
        norm_std: Mel normalization std
        head_dropout: Dropout probability before classification head
        backbone_mode: "frozen" | "lora" | "full"
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

        # Load pretrained backbone
        self.backbone = EfficientNetModel.from_pretrained(model_name)

        if backbone_mode == "frozen":
            self._freeze_backbone()
        elif backbone_mode == "lora":
            self._apply_lora(lora_rank, lora_alpha, lora_dropout, lora_targets)
        elif backbone_mode == "full":
            logger.info("Backbone fully unfrozen — all params trainable")
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        # Classification head
        embed_dim = self.backbone.config.hidden_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info("Backbone frozen — only head trains")

    def _apply_lora(self, rank, alpha, dropout, targets):
        """Apply LoRA adapters to the backbone via PEFT."""
        from peft import LoraConfig, get_peft_model

        if targets is None:
            # EfficientNet uses Conv1x1 projections in its MBConv blocks.
            # Target the squeeze-excite and projection layers.
            targets = [
                "project_conv",
                "expand_conv",
                "reduce",
                "expand",
            ]

        # Freeze backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"LoRA applied — {trainable:,} trainable / {total:,} total backbone params "
            f"({100*trainable/total:.1f}%)"
        )

    def train(self, mode=True):
        """Override to keep backbone BN stats in eval mode when frozen/lora."""
        super().train(mode)
        if self.backbone_mode in ("frozen", "lora"):
            # Keep batch norm in eval mode to use pretrained running stats
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
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
            mel = mel[:, :, : self.target_frames]
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
            padding_mask: unused, kept for interface compat
            spec_augment: Optional SpecAugment callable (training only)

        Returns:
            clip_logits: (B, C) clip-level raw logits
        """
        mel = self.compute_mel(waveforms)  # (B, 1, n_mels, T)

        if spec_augment is not None:
            mel = spec_augment(mel)

        if self.backbone_mode == "frozen":
            with torch.no_grad():
                outputs = self.backbone(pixel_values=mel)
                features = outputs.last_hidden_state
                pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            # LoRA or full — gradients flow through backbone
            outputs = self.backbone(pixel_values=mel)
            features = outputs.last_hidden_state
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)

        clip_logits = self.head(pooled)
        return clip_logits
