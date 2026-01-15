"""SSAMBA (Self-Supervised Audio Mamba) encoder wrapper.

Uses the official SSAMBA implementation from vendor/ssamba.
Requires the Vision Mamba (Vim) repository and mamba-ssm dependencies.

Reference: https://github.com/SiavashShams/ssamba
Paper: https://arxiv.org/abs/2405.11831
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from .base import AudioEncoder

# Vendor paths
VENDOR_SSAMBA_PATH = Path(__file__).parents[5] / "vendor" / "ssamba" / "src"
VENDOR_VIM_PATH = Path(__file__).parents[5] / "vendor" / "ssamba" / "Vim"


class SSAMBAEncoder(AudioEncoder):
    """SSAMBA encoder for audio feature extraction.

    Uses the Mamba state space model for efficient audio processing.
    Checkpoints available at:
    https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX

    Model sizes:
    - tiny: 7M params, 12 layers, 192 hidden dim
    - small: 26M params, 24 layers, 384 hidden dim
    - base: 99M params, 24 layers, 768 hidden dim

    Args:
        checkpoint_path: Path to SSAMBA checkpoint (.pth file)
        model_size: Model size ('tiny', 'small', 'base')
        fshape: Frequency dimension of patch
        tshape: Time dimension of patch
        fstride: Frequency stride for finetuning
        tstride: Time stride for finetuning
        input_fdim: Input frequency dimension (128 mel bins)
        input_tdim: Input time dimension
        device: Device to run model on
    """

    # Model configurations for different sizes
    MODEL_CONFIGS = {
        "tiny": {
            "embed_dim": 192,
            "depth": 12,
        },
        "small": {
            "embed_dim": 384,
            "depth": 24,
        },
        "base": {
            "embed_dim": 768,
            "depth": 24,
        },
    }

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "base",
        fshape: int = 16,
        tshape: int = 16,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 512,
        device: str = "cuda",
        norm_mean: float = -4.2677393,
        norm_std: float = 4.5689974,
    ):
        super().__init__()
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.model_size = model_size
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.fshape = fshape
        self.tshape = tshape
        self.fstride = fstride
        self.tstride = tstride

        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[model_size]
        self._embed_dim = config["embed_dim"]

        # Set up paths for vendor code
        self._setup_paths()

        # Load the model
        self.model = self._load_model(config)
        self.model.eval()
        self.model.to(device)

    def _setup_paths(self):
        """Add vendor paths to sys.path."""
        paths_to_add = [
            str(VENDOR_SSAMBA_PATH),
            str(VENDOR_VIM_PATH),
            str(VENDOR_VIM_PATH / "vim"),
            str(VENDOR_VIM_PATH / "vim" / "mamba-1p1p1"),
        ]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

    def _load_model(self, config):
        """Load the SSAMBA model from checkpoint."""
        try:
            from models.both_models import AMBAModel
        except ImportError as e:
            raise ImportError(
                "Failed to import SSAMBA model. Make sure the Vim repository is cloned "
                "inside vendor/ssamba/ and mamba-ssm is installed.\n"
                "See: https://github.com/SiavashShams/ssamba#installation\n"
                f"Original error: {e}"
            )

        # Vision Mamba config matching the pretrained checkpoints
        vision_mamba_config = {
            "img_size": (self.input_fdim, self.input_tdim),
            "patch_size": 16,
            "stride": 10,
            "embed_dim": config["embed_dim"],
            "depth": config["depth"],
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "final_pool_type": "mean",
            "if_abs_pos_embed": True,
            "if_rope": False,
            "if_rope_residual": False,
            "bimamba_type": "v2",
            "if_cls_token": True,
            "if_devide_out": True,
            "use_middle_cls_token": True,
        }

        # Load model in finetuning mode
        model = AMBAModel(
            label_dim=50,  # Placeholder
            fshape=self.fshape,
            tshape=self.tshape,
            fstride=self.fstride,
            tstride=self.tstride,
            input_fdim=self.input_fdim,
            input_tdim=self.input_tdim,
            model_size=self.model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path=str(self.checkpoint_path),
            vision_mamba_config=vision_mamba_config,
        )

        return model

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
            wav = wav - wav.mean()

            fbank = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.input_fdim,
                dither=0.0,
                frame_shift=10,
            )

            n_frames = fbank.shape[0]
            if n_frames < self.input_tdim:
                fbank = torch.nn.functional.pad(
                    fbank, (0, 0, 0, self.input_tdim - n_frames)
                )
            elif n_frames > self.input_tdim:
                fbank = fbank[: self.input_tdim, :]

            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            fbanks.append(fbank)

        return torch.stack(fbanks, dim=0)

    def _extract_features_internal(self, fbank: torch.Tensor) -> torch.Tensor:
        """Extract features from fbank without classification head.

        Args:
            fbank: (batch, time, freq) filter bank features

        Returns:
            Features of shape (batch, embed_dim)
        """
        # Import mamba ops if available
        try:
            from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
        except ImportError:
            RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

        x = fbank.unsqueeze(1).transpose(2, 3)  # (batch, 1, freq, time)

        B = x.shape[0]
        x = self.model.v.patch_embed(x)

        cls_tokens = self.model.v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.v.pos_embed
        x = self.model.v.pos_drop(x)

        # Mamba forward pass
        residual = None
        hidden_states = x

        if not self.model.v.if_bidirectional:
            for layer in self.model.v.layers:
                hidden_states, residual = layer(hidden_states, residual)
        else:
            for i in range(len(self.model.v.layers) // 2):
                hidden_states_f, residual_f = self.model.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.model.v.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.model.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.model.v.drop_path(hidden_states)
            hidden_states = self.model.v.norm_f(
                residual.to(dtype=self.model.v.norm_f.weight.dtype)
            )
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.model.v.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.model.v.drop_path(hidden_states),
                self.model.v.norm_f.weight,
                self.model.v.norm_f.bias,
                eps=self.model.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.model.v.residual_in_fp32,
            )

        x = self.model.v.norm_f(hidden_states)

        # Mean pool over all tokens except cls token
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
        waveform = waveform.squeeze(1)
        fbank = self._waveform_to_fbank(waveform).to(self.device)
        features = self._extract_features_internal(fbank)
        return features
