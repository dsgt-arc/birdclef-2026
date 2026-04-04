"""SSAMBA linear probe classifier with frozen pretrained backbone.

Loads backbone weights from an SSL pretrain checkpoint and trains
only a linear classification head. Uses single norm_f to match
the pretrain forward path (NOT forward_features which applies double norm).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.model import SSAMBAModel

logger = logging.getLogger(__name__)


class AttentionPoolingHead(nn.Module):
    """SED-style per-token classification with attention pooling.

    Each token gets an independent species prediction. At training time,
    predictions are aggregated via learned attention weights (which tokens
    matter for each species). At inference, framewise max captures brief calls.

    Adapted from exp 102's AttentionPoolingHead.

    Args:
        embed_dim: Input feature dimension per token
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

    def forward(self, token_features: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Args:
            token_features: (B, N, D) per-token backbone features
            padding_mask: (B, N) boolean, True for real tokens

        Returns:
            clip_probs: (B, C) clip-level probabilities
                Training: attention-weighted sum of per-token sigmoid probs
                Inference: framewise max of per-token sigmoid probs
        """
        frame_logits = self.frame_head(token_features)  # (B, N, C)

        if self.training:
            att_logits = self.att_head(token_features)  # (B, N, C)
            # Mask padded tokens with -inf before softmax
            if padding_mask is not None:
                att_logits = att_logits.masked_fill(
                    ~padding_mask.unsqueeze(-1), float("-inf"),
                )
            att_weights = torch.softmax(torch.tanh(att_logits), dim=1)  # (B, N, C)
            clip_probs = (att_weights * torch.sigmoid(frame_logits)).sum(dim=1)  # (B, C)
        else:
            if padding_mask is not None:
                # Mask padded tokens with -inf so they don't win the max
                frame_logits = frame_logits.masked_fill(
                    ~padding_mask.unsqueeze(-1), float("-inf"),
                )
            clip_probs = torch.sigmoid(frame_logits).max(dim=1).values  # (B, C)

        return clip_probs


class SSAMBALinearProbeClassifier(nn.Module):
    """Frozen SSAMBA backbone + attention pooling classification head.

    Accepts raw waveforms and computes mel spectrograms on GPU (via nnAudio),
    then runs the frozen backbone. Per-token predictions are aggregated via
    learned attention (train) or framewise max (inference).

    The forward pass replicates the pretrain model's single-norm path
    through the backbone, NOT SSAMBAModel.forward_features() which
    applies norm_f twice (a legacy AudioSet finetuning quirk).

    Args:
        config: SSAMBAConfig for the backbone
        mel_spec_layer: nnAudio MelSpectrogram module (runs on GPU)
        target_frames: Expected number of mel frames
        num_classes: Number of output classes
        head_dropout: Dropout before the linear head
    """

    def __init__(
        self,
        config: SSAMBAConfig,
        mel_spec_layer: nn.Module,
        target_frames: int,
        num_classes: int,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.config = config
        self.mel_spec = mel_spec_layer
        self.target_frames = target_frames
        self.backbone = SSAMBAModel(config)
        self.head = AttentionPoolingHead(config.embed_dim, num_classes, dropout=head_dropout)

    @torch.amp.autocast("cuda", enabled=False)
    def compute_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute AmplitudeToDB-normalized mel. Matches pretrain pipeline exactly."""
        waveforms = waveforms.float()
        mel = self.mel_spec(waveforms)  # (B, n_mels, T)

        mel = 10.0 * torch.log10(mel.clamp(min=1e-10))
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = torch.clamp(mel, min=mel_max - 80.0)
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        mel = (mel - mean) / std
        mel_min = mel.amin(dim=(-2, -1), keepdim=True)
        mel_max2 = mel.amax(dim=(-2, -1), keepdim=True)
        mel = (mel - mel_min) / (mel_max2 - mel_min + 1e-6)

        T_mel = mel.shape[2]
        if T_mel > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T_mel < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - T_mel), value=0)

        return mel.unsqueeze(1)  # (B, 1, n_mels, T)

    def forward(
        self,
        waveforms: torch.Tensor,
        real_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: waveform → mel (GPU) → backbone → head.

        Args:
            waveforms: (B, num_samples) raw audio
            real_samples: (B,) number of real samples per item for padding mask.
                If None, all positions treated as real.

        Returns:
            (B, num_classes) raw logits
        """
        B = waveforms.shape[0]
        mel_specs = self.compute_mel(waveforms)  # (B, 1, n_mels, T)

        # Compute padding mask from real_samples
        if real_samples is not None:
            hop_length = 320
            tpatch = self.config.tpatch
            fpatch = self.config.fpatch
            n_mels = self.config.img_size[0]
            t_dim = self.target_frames // tpatch
            f_dim = n_mels // fpatch
            num_tokens = f_dim * t_dim

            real_t_patches = (real_samples.float() / (hop_length * tpatch)).long().clamp(max=t_dim)
            # Freq-first order: token i → time_idx = i % t_dim
            indices = torch.arange(num_tokens, device=waveforms.device)
            padding_mask = indices.unsqueeze(0) % t_dim < real_t_patches.unsqueeze(1)  # (B, N)
        else:
            padding_mask = None

        # Patch embedding
        patch_embeddings = self.backbone.patch_embed(mel_specs)  # (B, N, D)
        num_patches = patch_embeddings.shape[1]

        # Add positional embedding (skip CLS position at index 0)
        x = patch_embeddings + self.backbone.pos_embed[:, 1:, :]

        # Insert CLS token
        if self.backbone.cls_token is not None:
            cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)

            if self.backbone.config.use_middle_cls_token:
                cls_pos = num_patches // 2
            else:
                cls_pos = 0

            if cls_pos == 0:
                x = torch.cat([cls_tokens, x], dim=1)
            else:
                x = torch.cat([x[:, :cls_pos, :], cls_tokens, x[:, cls_pos:, :]], dim=1)
        else:
            cls_pos = None

        x = self.backbone.pos_drop(x)

        # Mamba layers
        hidden_states, residual = self.backbone.layers(x)

        # Single norm_f (matches pretrain, NOT forward_features which does double norm)
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.backbone.drop_path(hidden_states)
        x = self.backbone.norm_f(residual.to(dtype=self.backbone.norm_f.weight.dtype))

        # Remove CLS token
        if cls_pos is not None:
            if cls_pos == 0:
                patch_repr = x[:, 1:, :]
            else:
                patch_repr = torch.cat([x[:, :cls_pos, :], x[:, cls_pos + 1:, :]], dim=1)
        else:
            patch_repr = x

        # Per-token attention pooling head
        return self.head(patch_repr, padding_mask=padding_mask)

    @classmethod
    def from_pretrain_checkpoint(
        cls,
        ckpt_path: str,
        config: SSAMBAConfig,
        mel_spec_layer: nn.Module,
        target_frames: int,
        num_classes: int,
        head_dropout: float = 0.3,
    ) -> "SSAMBALinearProbeClassifier":
        """Create classifier with backbone weights from an SSL pretrain checkpoint.

        Args:
            ckpt_path: Path to Lightning checkpoint from PretrainLightningModule
            config: SSAMBAConfig matching the pretrained model
            mel_spec_layer: nnAudio MelSpectrogram module
            target_frames: Expected number of mel frames
            num_classes: Number of output classes
            head_dropout: Dropout before linear head

        Returns:
            Classifier with frozen backbone and randomly initialized head
        """
        model = cls(config, mel_spec_layer, target_frames, num_classes, head_dropout)

        # Load checkpoint and extract backbone weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        backbone_sd = {}
        for key, val in state_dict.items():
            if key.startswith("model.backbone."):
                backbone_key = key[len("model."):]  # "backbone.xxx"
                backbone_sd[backbone_key] = val

        missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
        if missing:
            logger.warning(f"Missing backbone keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected backbone keys: {unexpected}")

        loaded = len(backbone_sd)
        logger.info(f"Loaded {loaded} backbone parameters from {ckpt_path}")

        # Freeze backbone
        frozen = 0
        for param in model.backbone.parameters():
            param.requires_grad = False
            frozen += 1

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Frozen backbone: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")

        return model
