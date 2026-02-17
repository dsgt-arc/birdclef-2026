"""Frame-tokenized ViT for audio classification.

Experiment 004: Replaces standard 16x16 square patches with vertical
frame patches (128xW) that span the full frequency axis. Isolated from
AudioClassifier to keep experiment-specific changes self-contained.

Uses nnAudio for GPU-accelerated mel spectrogram computation — the model
accepts raw waveforms (batch, num_samples) and computes mel on-device.

Patch configurations:
    - Baseline:  (16, 16)  -> 8x31 = 248 tokens, embed_input = 256
    - Config A:  (128, 2)  -> 1x250 = 250 tokens, embed_input = 256
    - Config B:  (128, 4)  -> 1x125 = 125 tokens, embed_input = 512
    - Config C:  (128, 1)  -> 1x500 = 500 tokens, embed_input = 128
"""

import timm
import torch
import torch.nn as nn

from birdset_benchmark.preprocessing.nnaudio_mel import NNAudioMelSpectrogram


class FrameTokenizedViT(nn.Module):
    """ViT with configurable patch tokenization for audio spectrograms.

    Accepts raw waveforms (batch, num_samples) and computes mel spectrograms
    on GPU via nnAudio before passing through the ViT backbone.

    Args:
        model_name: timm model name (e.g., 'vit_small_patch16_224')
        num_classes: Number of output classes
        img_size: Input image size as (height, width), default (128, 500)
        patch_size: Patch size as (freq, time), default (16, 16)
        pretrained: Whether to use pretrained weights (default: False)
        drop_rate: Dropout rate (default: 0.0)
        sample_rate: Audio sample rate for mel computation (default: 32000)
        n_mels: Number of mel bins (default: 128)
        frame_shift_ms: Frame shift in ms (default: 10)
        target_frames: Target mel frames (default: 500)
        norm_mean: Mel normalization mean (default: -4.268)
        norm_std: Mel normalization std (default: 4.569)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        img_size: tuple[int, int] = (128, 500),
        patch_size: tuple[int, int] = (16, 16),
        pretrained: bool = False,
        drop_rate: float = 0.0,
        sample_rate: int = 32000,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 500,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        super().__init__()

        self.patch_size = patch_size

        # GPU-accelerated mel spectrogram front-end
        self.mel_spec = NNAudioMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            frame_shift_ms=frame_shift_ms,
            target_frames=target_frames,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # Create backbone with custom patch_size
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            img_size=img_size,
            patch_size=patch_size,
        )

        # Adapt input from 3 channels to 1 channel
        self._adapt_input_channels(pretrained)

        # Detect embedding dimension
        if hasattr(self.backbone, "embed_dim"):
            embed_dim = self.backbone.embed_dim
        else:
            raise ValueError(f"Cannot detect embed_dim for model: {model_name}")

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _adapt_input_channels(self, pretrained: bool):
        """Replace 3-channel patch embedding with 1-channel variant."""
        old_conv = self.backbone.patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
        self.backbone.patch_embed.proj = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from waveforms or pre-computed mels to logits.

        Args:
            x: Raw waveforms (batch, num_samples) or
               pre-computed mels (batch, 1, n_mels, target_frames)

        Returns:
            Logits, shape (batch, num_classes)
        """
        if x.dim() == 2:
            # Disable AMP autocast for mel computation — nnAudio's Conv1d STFT
            # loses spectral fidelity in float16, causing a 2x cmAP gap vs torchaudio
            with torch.amp.autocast(device_type="cuda", enabled=False):
                x = self.mel_spec(x.float())
        # else: x is already pre-computed mel (batch, 1, n_mels, target_frames)
        features = self.backbone(x)
        logits = self.head(features)
        return logits
