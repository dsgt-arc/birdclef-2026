"""SSAST (Self-Supervised Audio Spectrogram Transformer) encoder wrapper.

Uses the official SSAST implementation from vendor/ssast.
Reference: https://github.com/YuanGongND/ssast
"""

import sys
from pathlib import Path

import torch
import torchaudio

from .base import AudioEncoder

# Add vendor SSAST to path
VENDOR_SSAST_PATH = Path(__file__).parents[5] / "vendor" / "ssast" / "src"


class SSASTEncoder(AudioEncoder):
    """SSAST encoder for audio feature extraction.

    Uses the self-supervised pre-trained AST model from the SSAST paper.
    Checkpoints can be downloaded from:
    https://www.dropbox.com/sh/0zcou1nm9i9mg7l/AAB73D8mJ3cJPaHDQCIHrH5Ra?dl=0

    Args:
        checkpoint_path: Path to SSAST checkpoint (.pth file)
        model_size: Model size ('tiny', 'small', 'base', 'base_nokd')
        fshape: Frequency dimension of patch (16 for patch-based, 128 for frame-based)
        tshape: Time dimension of patch (16 for patch-based, 2 for frame-based)
        fstride: Frequency stride for finetuning (typically 10 for overlap)
        tstride: Time stride for finetuning (typically 10 for overlap)
        input_fdim: Input frequency dimension (128 mel bins)
        input_tdim: Input time dimension (depends on audio length)
        device: Device to run model on
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "base",
        fshape: int = 16,
        tshape: int = 16,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 512,  # ~5 seconds at 16kHz with 10ms hop
        device: str = "cuda",
        norm_mean: float = -4.2677393,
        norm_std: float = 4.5689974,
    ):
        super().__init__()
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Add vendor path to load the model
        if str(VENDOR_SSAST_PATH) not in sys.path:
            sys.path.insert(0, str(VENDOR_SSAST_PATH))

        # Import SSAST model (requires timm==0.4.5)
        from models.ast_models import ASTModel as SSASTModel

        # Load model in finetuning mode (pretrain_stage=False)
        self.model = SSASTModel(
            label_dim=50,  # Placeholder, we extract features before MLP head
            fshape=fshape,
            tshape=tshape,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            model_size=model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path=str(checkpoint_path),
        )

        self.model.eval()
        self.model.to(device)

        # SSAST-base has 768 hidden dim
        self._embed_dim = self.model.original_embedding_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to filter bank features.

        Args:
            waveform: (batch, samples) at 16kHz

        Returns:
            Filter bank features of shape (batch, input_tdim, input_fdim)
        """
        batch_size = waveform.shape[0]
        fbanks = []

        for i in range(batch_size):
            wav = waveform[i]
            # Normalize waveform
            wav = wav - wav.mean()

            # Compute filter bank using Kaldi
            fbank = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.input_fdim,
                dither=0.0,
                frame_shift=10,
            )  # (time, input_fdim)

            # Pad or truncate to target length
            n_frames = fbank.shape[0]
            if n_frames < self.input_tdim:
                fbank = torch.nn.functional.pad(
                    fbank, (0, 0, 0, self.input_tdim - n_frames)
                )
            elif n_frames > self.input_tdim:
                fbank = fbank[: self.input_tdim, :]

            # Normalize
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

            fbanks.append(fbank)

        return torch.stack(fbanks, dim=0)  # (batch, time, freq)

    def _extract_features_internal(self, fbank: torch.Tensor) -> torch.Tensor:
        """Extract features from fbank without classification head.

        This reimplements the forward pass to get features before MLP head.

        Args:
            fbank: (batch, time, freq) filter bank features

        Returns:
            Features of shape (batch, embed_dim)
        """
        # SSAST expects input as (batch, time, freq)
        # Then converts to (batch, 1, freq, time) internally
        x = fbank.unsqueeze(1)  # (batch, 1, time, freq)
        x = x.transpose(2, 3)  # (batch, 1, freq, time)

        B = x.shape[0]
        x = self.model.v.patch_embed(x)

        # Add cls token(s)
        if self.model.cls_token_num == 2:
            cls_tokens = self.model.v.cls_token.expand(B, -1, -1)
            dist_token = self.model.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.model.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.model.v.pos_embed
        x = self.model.v.pos_drop(x)

        # Pass through transformer blocks
        for blk in self.model.v.blocks:
            x = blk(x)
        x = self.model.v.norm(x)

        # Average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.model.cls_token_num :, :], dim=1)

        return x

    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio tensor of shape (batch, 1, num_samples) at 16kHz

        Returns:
            Features of shape (batch, embed_dim) - mean pooled over time
        """
        # Remove channel dimension: (batch, 1, samples) -> (batch, samples)
        waveform = waveform.squeeze(1)

        # Convert to filter bank features
        fbank = self._waveform_to_fbank(waveform).to(self.device)

        # Extract features (before MLP head)
        features = self._extract_features_internal(fbank)

        return features
