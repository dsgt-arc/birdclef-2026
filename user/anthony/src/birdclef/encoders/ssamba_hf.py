"""SSAMBA HuggingFace port encoder wrapper.

Uses the HuggingFace-compatible implementation from experiment 004 that doesn't
require CUDA kernels (mamba-ssm). This allows running SSAMBA on CPU or GPU
without the custom CUDA compilation.

Reference: https://github.com/SiavashShams/ssamba
Paper: https://arxiv.org/abs/2405.11831
"""

import sys
from pathlib import Path

import torch
import torchaudio

from .base import AudioEncoder

# Path to the experiment 004 HF port
# __file__ is .../user/anthony/src/birdclef/encoders/ssamba_hf.py
# parents[3] is .../user/anthony
SSAMBA_HF_DIR = Path(__file__).parents[3] / "experiments" / "004-ssamba-hf-port" / "ssamba_hf"


class SSAMBAHFEncoder(AudioEncoder):
    """SSAMBA encoder using HuggingFace-compatible implementation.

    This encoder uses the pure PyTorch implementation from experiment 004 that
    replaces the CUDA-only mamba_ssm kernels with HuggingFace's transformers.
    This allows running SSAMBA on CPU or any GPU without custom CUDA compilation.

    Checkpoints available at:
    https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX

    Model sizes:
    - tiny: 7M params, 192 hidden dim
    - small: 26M params, 384 hidden dim
    - base: 99M params, 768 hidden dim

    Args:
        checkpoint_path: Path to SSAMBA checkpoint (.pth file)
        model_size: Model size ('tiny', 'small', 'base') - only used for logging
        input_fdim: Input frequency dimension (128 mel bins)
        input_tdim: Input time dimension (512 frames for 5 seconds)
        device: Device to run model on
        norm_mean: Mean for normalization (AudioSet statistics)
        norm_std: Std for normalization (AudioSet statistics)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "small",
        input_fdim: int = 128,
        input_tdim: int = 512,
        device: str = "cpu",
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

        # Add experiment 004 ssamba_hf to path
        if str(SSAMBA_HF_DIR) not in sys.path:
            sys.path.insert(0, str(SSAMBA_HF_DIR))

        # Import from the HF port
        from ssamba_hf import load_from_ssamba_checkpoint

        # Load model from checkpoint
        self.model, self.config = load_from_ssamba_checkpoint(
            str(checkpoint_path),
            device=device,
            verbose=True,
        )
        self.model.eval()

        self._embed_dim = self.config.embed_dim

        # Get the model's expected input dimensions from config
        # config.img_size is (freq_dim, time_dim)
        self._model_fdim, self._model_tdim = self.config.img_size
        print(f"Model expects input: ({self._model_tdim} frames, {self._model_fdim} freq bins)")

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to filter bank features.

        Uses Kaldi fbank with 128 mel bins and 10ms frame shift,
        matching the original SSAST/SSAMBA preprocessing.

        The output is padded/truncated to match the model's expected
        input size (from checkpoint config).

        Args:
            waveform: (batch, samples) at 16kHz

        Returns:
            Filter bank features of shape (batch, model_tdim, model_fdim)
        """
        batch_size = waveform.shape[0]
        fbanks = []

        # Use model's expected dimensions
        target_tdim = self._model_tdim
        target_fdim = self._model_fdim

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
                num_mel_bins=target_fdim,
                dither=0.0,
                frame_shift=10,
            )  # (time, target_fdim)

            # Pad or truncate to model's expected length
            n_frames = fbank.shape[0]
            if n_frames < target_tdim:
                fbank = torch.nn.functional.pad(
                    fbank, (0, 0, 0, target_tdim - n_frames)
                )
            elif n_frames > target_tdim:
                fbank = fbank[:target_tdim, :]

            # Normalize with AudioSet statistics
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

            fbanks.append(fbank)

        return torch.stack(fbanks, dim=0)  # (batch, time, freq)

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

        # Extract embeddings using the HF model
        # Input: (batch, time, freq)
        embeddings = self.model.get_embeddings(fbank)

        return embeddings

    @torch.no_grad()
    def extract_patch_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Per-patch (un-pooled) features for prototypical probing.

        Returns the patch token grid with the (middle) CLS token dropped, shape
        (batch, num_patches, embed_dim). Used by exp 300's `--patch` cache for the
        prototypical probe's frame-level (Strategy B) max-similarity readout.
        """
        waveform = waveform.squeeze(1)
        fbank = self._waveform_to_fbank(waveform).to(self.device)
        # Mirror forward(): (B, T, F) -> (B, 1, F, T) before forward_features.
        x = fbank.unsqueeze(1).transpose(2, 3)
        tokens = self.model.forward_features(x, return_all_tokens=True)  # (B, N+1, D)
        cls_pos = getattr(self.model, "_cls_token_position", 0)
        if cls_pos == 0:
            return tokens[:, 1:, :]
        return torch.cat([tokens[:, :cls_pos, :], tokens[:, cls_pos + 1:, :]], dim=1)
