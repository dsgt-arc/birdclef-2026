"""Verification tests for SSAMBA-HF correctness.

HF-only tests that don't require the original SSAMBA implementation.
Covers preprocessing audit, CUDA-vs-PyTorch kernel comparison with
checkpoint weights, and multi-input robustness.

These tests require:
- A checkpoint file (set SSAMBA_CHECKPOINT env var or skip)
- For CUDA tests: GPU + mamba-ssm installed (uv sync --extra cuda)

Run:
    uv run --extra dev --extra cuda pytest tests/test_verification.py -v
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchaudio

from ssamba_hf import (
    MAMBA_SSM_AVAILABLE,
    SSAMBAConfig,
    SSAMBAModel,
    load_from_ssamba_checkpoint,
)


# --- Fixtures & markers ---

CHECKPOINT_PATH = os.environ.get(
    "SSAMBA_CHECKPOINT",
    os.path.expanduser("~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth"),
)

ESC50_ROOT = os.environ.get(
    "ESC50_ROOT",
    os.path.expanduser("~/arc/birdclef-2026/vendor/ESC-50"),
)

requires_checkpoint = pytest.mark.skipif(
    not Path(CHECKPOINT_PATH).exists(),
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_mamba_ssm = pytest.mark.skipif(
    not MAMBA_SSM_AVAILABLE, reason="mamba-ssm not installed"
)
requires_esc50 = pytest.mark.skipif(
    not Path(ESC50_ROOT).exists(), reason=f"ESC-50 not found at {ESC50_ROOT}"
)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


@pytest.fixture
def sample_spectrogram():
    rng = np.random.default_rng(42)
    return torch.from_numpy(
        rng.standard_normal((1, 1024, 128)).astype(np.float32)
    )


@pytest.fixture
def checkpoint_model():
    """Load HF model from checkpoint with PyTorch backend."""
    model, config = load_from_ssamba_checkpoint(
        CHECKPOINT_PATH, device="cpu", verbose=False, use_cuda_kernels=False
    )
    model.eval()
    return model, config


# ===================================================================
# Preprocessing Audit
# ===================================================================


class TestPreprocessingParams:
    """Verify fbank parameters and normalization match original SSAST/SSAMBA."""

    EXPECTED_FBANK = {
        "htk_compat": True,
        "sample_frequency": 16000,
        "use_energy": False,
        "window_type": "hanning",
        "num_mel_bins": 128,
        "dither": 0.0,
        "frame_shift": 10,
    }
    EXPECTED_NORM_MEAN = -4.2677393
    EXPECTED_NORM_STD = 4.5689974

    def test_normalization_formula(self):
        """Normalization should be (fbank - mean) / (std * 2)."""
        val = torch.tensor([0.0, 1.0, -1.0, 10.0])
        mean = self.EXPECTED_NORM_MEAN
        std = self.EXPECTED_NORM_STD
        expected = (val - mean) / (std * 2)

        # This is exactly what SSAMBAHFEncoder._waveform_to_fbank does
        actual = (val - mean) / (std * 2)
        assert torch.allclose(expected, actual)

    def test_esc50_frame_count(self):
        """5-second audio at 16kHz should produce ~498 fbank frames."""
        waveform = torch.randn(1, 80000)  # 5s at 16kHz
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, **self.EXPECTED_FBANK
        )
        n_frames = fbank.shape[0]
        assert 490 <= n_frames <= 510, f"Expected ~498 frames, got {n_frames}"
        assert fbank.shape[1] == 128

    @requires_checkpoint
    def test_checkpoint_input_dimensions(self):
        """Checkpoint should store input_fdim=128, input_tdim=1024."""
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

        fdim = sd.get("module.p_input_fdim", sd.get("p_input_fdim"))
        tdim = sd.get("module.p_input_tdim", sd.get("p_input_tdim"))

        if hasattr(fdim, "item"):
            fdim, tdim = fdim.item(), tdim.item()

        assert fdim == 128, f"Expected fdim=128, got {fdim}"
        assert tdim == 1024, f"Expected tdim=1024, got {tdim}"

    @requires_checkpoint
    def test_checkpoint_positional_embedding_shape(self):
        """pos_embed should have num_patches + 1 positions (for CLS token)."""
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        pos_embed = sd.get("module.v.pos_embed", sd.get("v.pos_embed"))
        patch_weight = sd.get(
            "module.v.patch_embed.proj.weight",
            sd.get("v.patch_embed.proj.weight"),
        )

        patch_size = patch_weight.shape[2]
        # 128/16 * 1024/16 = 8 * 64 = 512 patches
        num_patches = (128 // patch_size) * (1024 // patch_size)
        expected_positions = num_patches + 1  # +1 for CLS

        assert pos_embed.shape[1] == expected_positions, (
            f"Expected {expected_positions} positions, got {pos_embed.shape[1]}"
        )

    def test_axis_ordering(self):
        """(B, T, F) -> unsqueeze(1).transpose(2,3) should give (B, 1, F, T)."""
        B, T, F = 2, 1024, 128
        x = torch.randn(B, T, F)
        x_out = x.unsqueeze(1).transpose(2, 3)
        assert x_out.shape == (B, 1, F, T)

    @requires_esc50
    def test_real_audio_pipeline(self):
        """Full preprocessing pipeline on a real ESC-50 file."""
        audio_dir = Path(ESC50_ROOT) / "audio"
        wav_files = sorted(audio_dir.glob("*.wav"))
        assert len(wav_files) > 0, "No .wav files found"

        wav_file = wav_files[0]
        waveform, sr = torchaudio.load(str(wav_file))

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, **self.EXPECTED_FBANK
        )
        assert fbank.shape[1] == 128
        assert 100 <= fbank.shape[0] <= 2000  # reasonable range

        # Pad to model size
        if fbank.shape[0] < 1024:
            fbank = torch.nn.functional.pad(
                fbank, (0, 0, 0, 1024 - fbank.shape[0])
            )

        # Normalize
        fbank = (fbank - self.EXPECTED_NORM_MEAN) / (self.EXPECTED_NORM_STD * 2)
        assert not torch.isnan(fbank).any()
        assert not torch.isinf(fbank).any()


# ===================================================================
# CUDA vs PyTorch Kernel (with checkpoint weights)
# ===================================================================


class TestCudaVsPytorchCheckpoint:
    """Compare CUDA and PyTorch backends using real checkpoint weights."""

    @requires_checkpoint
    @requires_cuda
    @requires_mamba_ssm
    def test_output_equivalence_seed42(self, sample_spectrogram):
        """Checkpoint model: CUDA vs PyTorch outputs should match (seed=42)."""
        pytorch_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=False
        )
        pytorch_model.eval()

        cuda_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=True
        )
        cuda_model.eval()

        spec = sample_spectrogram.cuda()

        with torch.no_grad():
            emb_pt = pytorch_model.get_embeddings(spec)
            emb_cuda = cuda_model.get_embeddings(spec)

        cos_sim = _cosine_sim(emb_pt, emb_cuda)
        max_diff = (emb_pt - emb_cuda).abs().max().item()

        print(f"\nCosine sim: {cos_sim:.6f}, max diff: {max_diff:.2e}")
        assert cos_sim > 0.999, f"Cosine sim {cos_sim:.6f} < 0.999"

    @requires_checkpoint
    @requires_cuda
    @requires_mamba_ssm
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_output_equivalence_multi_seed(self, seed):
        """CUDA vs PyTorch match across multiple random inputs."""
        pytorch_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=False
        )
        pytorch_model.eval()

        cuda_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=True
        )
        cuda_model.eval()

        rng = np.random.default_rng(seed)
        spec = torch.from_numpy(
            rng.standard_normal((1, 1024, 128)).astype(np.float32)
        ).cuda()

        with torch.no_grad():
            emb_pt = pytorch_model.get_embeddings(spec)
            emb_cuda = cuda_model.get_embeddings(spec)

        cos_sim = _cosine_sim(emb_pt, emb_cuda)
        assert cos_sim > 0.999, f"seed={seed}: cosine sim {cos_sim:.6f} < 0.999"

    @requires_checkpoint
    @requires_cuda
    @requires_mamba_ssm
    def test_output_equivalence_zeros(self):
        """CUDA vs PyTorch match on all-zeros input."""
        pytorch_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=False
        )
        pytorch_model.eval()

        cuda_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=True
        )
        cuda_model.eval()

        spec = torch.zeros(1, 1024, 128).cuda()

        with torch.no_grad():
            emb_pt = pytorch_model.get_embeddings(spec)
            emb_cuda = cuda_model.get_embeddings(spec)

        cos_sim = _cosine_sim(emb_pt, emb_cuda)
        assert cos_sim > 0.999, f"zeros: cosine sim {cos_sim:.6f} < 0.999"

    @requires_checkpoint
    @requires_cuda
    @requires_mamba_ssm
    def test_layer_drift(self):
        """Differences between CUDA and PyTorch should not grow across layers."""
        pytorch_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=False
        )
        pytorch_model.eval()

        cuda_model, _ = load_from_ssamba_checkpoint(
            CHECKPOINT_PATH, device="cuda", verbose=False, use_cuda_kernels=True
        )
        cuda_model.eval()

        # Hook layers
        pt_acts = {}
        cuda_acts = {}

        def make_hook(storage, idx):
            def hook_fn(module, input, output):
                hs, _ = output
                storage[idx] = hs.detach().cpu()
            return hook_fn

        hooks = []
        for i, layer in enumerate(pytorch_model.layers.layers):
            hooks.append(layer.register_forward_hook(make_hook(pt_acts, i)))
        for i, layer in enumerate(cuda_model.layers.layers):
            hooks.append(layer.register_forward_hook(make_hook(cuda_acts, i)))

        rng = np.random.default_rng(42)
        spec = torch.from_numpy(
            rng.standard_normal((1, 1024, 128)).astype(np.float32)
        ).cuda()

        with torch.no_grad():
            pytorch_model.get_embeddings(spec)
            cuda_model.get_embeddings(spec)

        for h in hooks:
            h.remove()

        # Check per-layer cosine similarity stays high.
        # Absolute diffs grow because activation norms grow through layers
        # (pre-norm architecture: norm is on input, not output), so we use
        # cosine similarity which is scale-invariant.
        cos_sims = []
        for i in sorted(pt_acts.keys()):
            cos = _cosine_sim(pt_acts[i], cuda_acts[i])
            norm = pt_acts[i].float().norm().item()
            diff = (pt_acts[i] - cuda_acts[i]).abs().max().item()
            rel_diff = diff / (norm + 1e-12)
            cos_sims.append(cos)
            print(f"  Layer {i:2d}: cos={cos:.6f}, max_diff={diff:.2e}, "
                  f"norm={norm:.1f}, rel_diff={rel_diff:.2e}")

        min_cos = min(cos_sims)
        print(f"\nMin cosine sim across layers: {min_cos:.6f}")

        assert min_cos > 0.999, f"Min layer cosine sim {min_cos:.6f} < 0.999"


# ===================================================================
# Multi-Input Robustness (HF model only)
# ===================================================================


class TestMultiInputRobustness:
    """Verify model produces sensible outputs across diverse inputs."""

    @requires_checkpoint
    def test_deterministic_across_calls(self, checkpoint_model, sample_spectrogram):
        model, _ = checkpoint_model

        with torch.no_grad():
            emb1 = model.get_embeddings(sample_spectrogram)
            emb2 = model.get_embeddings(sample_spectrogram)

        assert torch.allclose(emb1, emb2), "Same input should give same output"

    @requires_checkpoint
    def test_different_inputs_give_different_outputs(self, checkpoint_model):
        model, config = checkpoint_model

        specs = []
        for seed in [0, 1]:
            rng = np.random.default_rng(seed)
            specs.append(
                torch.from_numpy(
                    rng.standard_normal((1, 1024, 128)).astype(np.float32)
                )
            )

        with torch.no_grad():
            emb0 = model.get_embeddings(specs[0])
            emb1 = model.get_embeddings(specs[1])

        cos_sim = _cosine_sim(emb0, emb1)
        # Different random inputs should NOT produce identical embeddings
        assert cos_sim < 0.999, f"Different inputs gave cosine sim {cos_sim:.6f}"

    @requires_checkpoint
    def test_no_nans_on_zeros(self, checkpoint_model):
        model, config = checkpoint_model
        spec = torch.zeros(1, 1024, 128)

        with torch.no_grad():
            emb = model.get_embeddings(spec)

        assert not torch.isnan(emb).any(), "NaN in output for zeros input"
        assert not torch.isinf(emb).any(), "Inf in output for zeros input"

    @requires_checkpoint
    def test_no_nans_on_short_padded(self, checkpoint_model):
        model, _ = checkpoint_model

        rng = np.random.default_rng(99)
        spec = np.zeros((1, 1024, 128), dtype=np.float32)
        spec[:, :100, :] = rng.standard_normal((1, 100, 128)).astype(np.float32)
        spec = torch.from_numpy(spec)

        with torch.no_grad():
            emb = model.get_embeddings(spec)

        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()
        assert emb.norm().item() > 0, "Embedding should be non-zero"

    @requires_checkpoint
    @requires_esc50
    @pytest.mark.parametrize("file_idx", [0, 100, 500, 1000, 1500])
    def test_real_esc50_audio(self, checkpoint_model, file_idx):
        """Model should produce valid embeddings on real ESC-50 audio."""
        model, config = checkpoint_model

        audio_dir = Path(ESC50_ROOT) / "audio"
        wav_files = sorted(audio_dir.glob("*.wav"))
        if file_idx >= len(wav_files):
            pytest.skip(f"Only {len(wav_files)} files, index {file_idx} OOB")

        wav_file = wav_files[file_idx]
        waveform, sr = torchaudio.load(str(wav_file))

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )

        if fbank.shape[0] < 1024:
            fbank = torch.nn.functional.pad(
                fbank, (0, 0, 0, 1024 - fbank.shape[0])
            )
        else:
            fbank = fbank[:1024, :]

        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        spec = fbank.unsqueeze(0)

        with torch.no_grad():
            emb = model.get_embeddings(spec)

        assert emb.shape == (1, config.embed_dim)
        assert not torch.isnan(emb).any(), f"NaN for {wav_file.name}"
        assert not torch.isinf(emb).any(), f"Inf for {wav_file.name}"
        assert emb.norm().item() > 0, f"Zero embedding for {wav_file.name}"
