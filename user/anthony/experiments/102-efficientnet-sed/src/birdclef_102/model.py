"""Backbone-agnostic classifier with optional LoRA adapters and configurable pooling.

Supports backbones: EfficientNet-B1, ConvNeXt-Base (auto-detected from HF config).

Supports three backbone modes:
  - frozen: backbone frozen, only head trains
  - lora: backbone frozen + LoRA adapters
  - full: backbone fully unfrozen

Supports two pooling modes:
  - avg: global average pooling (simple, fast)
  - attention: attention-weighted pooling (train) / framewise max (inference)

Architecture:
    waveform → nnAudio mel → SpecAugment (train)
    → backbone (+ optional LoRA)
    → mean over freq → (B, D, T) frame features
    → pooling → (B, C) clip logits
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

# Default LoRA targets per backbone architecture
LORA_TARGETS = {
    "efficientnet": ["project_conv", "expand_conv", "reduce", "expand"],
    "convnext": ["pwconv1", "pwconv2"],  # HF transformers names (timm uses fc1/fc2)
}

def _get_backbone_type(model_name: str) -> str:
    """Detect backbone type from HuggingFace config."""
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "").lower()
    if "efficientnet" in model_type:
        return "efficientnet"
    elif "convnext" in model_type:
        return "convnext"
    else:
        raise ValueError(f"Unsupported model_type: {model_type} for {model_name}")

def _get_embed_dim(backbone, backbone_type: str) -> int:
    """Get the output feature dimension from backbone config."""
    if backbone_type == "efficientnet":
        return backbone.config.hidden_dim  # 1280
    elif backbone_type == "convnext":
        return backbone.config.hidden_sizes[-1]  # 1024
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")


class AttentionPoolingHead(nn.Module):
    """SED-style attention pooling + frame classification head.

    Training: attention-weighted mean of frame logits
    Inference: framewise max of frame logits

    Optionally includes a hidden projection layer (GEMM) before the
    attention/frame heads for additional capacity.

    Args:
        embed_dim: Input feature dimension (1280 for EfficientNet-B1)
        num_classes: Number of output classes
        hidden_dim: If > 0, add a Linear + ReLU projection before heads
        dropout: Dropout before linear layers
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_dim = hidden_dim
        else:
            self.projection = None
            head_dim = embed_dim

        self.frame_head = nn.Sequential(
            nn.Dropout(dropout) if hidden_dim == 0 else nn.Identity(),
            nn.Linear(head_dim, num_classes),
        )
        self.att_head = nn.Sequential(
            nn.Dropout(dropout) if hidden_dim == 0 else nn.Identity(),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (B, T, D) per-frame features

        Returns:
            clip_probs: (B, C) clip-level probabilities (not logits!)
                Training: attention-weighted mean of per-frame sigmoid probs
                Inference: framewise max of per-frame sigmoid probs
        """
        if self.projection is not None:
            frame_features = self.projection(frame_features)  # (B, T, hidden_dim)

        frame_logits = self.frame_head(frame_features)  # (B, T, C)

        if self.training:
            # Standard SED formula (PANNs/AttBlockV2 pattern):
            # clip = sum(softmax(tanh(att)) * sigmoid(frame), dim=T)
            # tanh bounds attention logits to [-1,1], preventing peaky softmax
            att_logits = self.att_head(frame_features)  # (B, T, C)
            att_weights = torch.softmax(torch.tanh(att_logits), dim=1)  # (B, T, C)
            clip_probs = (att_weights * torch.sigmoid(frame_logits)).sum(dim=1)  # (B, C)
        else:
            # Framewise max of sigmoid probs: catches brief calls
            clip_probs = torch.sigmoid(frame_logits).max(dim=1).values  # (B, C)

        return clip_probs


class EfficientNetClassifier(nn.Module):
    """Backbone-agnostic classifier with configurable fine-tuning and pooling.

    Supports EfficientNet-B1 and ConvNeXt-Base backbones from HuggingFace.
    Backbone type is auto-detected from the model config.

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
        head_hidden_dim: int = 0,
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

        # Auto-detect and load pretrained backbone
        self.backbone_type = _get_backbone_type(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        embed_dim = _get_embed_dim(self.backbone, self.backbone_type)
        logger.info(f"Backbone: {self.backbone_type}, embed_dim: {embed_dim}")

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
            self.head = AttentionPoolingHead(
                embed_dim, num_classes, hidden_dim=head_hidden_dim, dropout=head_dropout,
            )
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
            targets = LORA_TARGETS.get(self.backbone_type, LORA_TARGETS["efficientnet"])

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

    @torch.amp.autocast("cuda", enabled=False)
    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram in fp32 — nnAudio FFT is not fp16-safe."""
        waveforms = waveforms.float()
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
            features: (B, D, H, W) backbone feature map
                D=1280 for EfficientNet-B1, D=1024 for ConvNeXt-Base
        """
        if self.backbone_mode == "frozen":
            with torch.no_grad():
                return self.backbone(pixel_values=mel).last_hidden_state
        else:
            return self.backbone(pixel_values=mel).last_hidden_state

    def forward(
        self,
        waveforms: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        spec_augment=None,
        mel: torch.Tensor | None = None,
    ):
        """Forward pass to clip-level predictions.

        Accepts either raw waveforms (mel computed on GPU) or pre-computed mel
        spectrograms (mel computed on CPU in dataloader for better throughput).

        Args:
            waveforms: (B, num_samples) raw audio — used if mel is None
            padding_mask: unused, kept for interface compat
            spec_augment: Optional SpecAugment callable (training only)
            mel: (B, 1, n_mels, T) pre-computed mel — skips GPU mel computation

        Returns:
            avg pooling:       (B, C) raw logits — apply sigmoid for probabilities
            attention pooling: (B, C) probabilities in [0, 1] — already sigmoided
        """
        if mel is None:
            mel = self.compute_mel(waveforms)

        if spec_augment is not None:
            mel = spec_augment(mel)

        features = self._extract_features(mel)  # (B, D, H, W)

        if self.pooling_mode == "avg":
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, D)
            return self.head(pooled)  # logits
        elif self.pooling_mode == "attention":
            frame_features = features.mean(dim=2).permute(0, 2, 1)  # (B, T, D)
            return self.head(frame_features)  # probabilities
