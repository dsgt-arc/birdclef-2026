"""Test that SEDWrapper handles AttHead's torch.logit without NaN propagation.

Background: AttHead computes clipwise via torch.logit(clamp(sigmoid(...))),
which can produce NaN in ONNX. With blend_clip=0.0, the old wrapper still
computed clipwise and multiplied by 0.0 — but 0.0 * NaN = NaN (IEEE 754),
corrupting the entire output. This caused v6k to score 0.508 LB (random).

The fix: skip clipwise computation entirely when blend_clip == 0.0.

Usage:
    cd user/murilo
    python -m pytest tests/test_sed_wrapper_nan.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add user src to path
USER_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, USER_SRC_DIR)

from birdclef.heads.sed_head import AttHead, LSEHead  # noqa: E402

SAMPLE_AUDIO = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_audio/rubthr1/iNat1022679.ogg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_real_mel(audio_path: str = SAMPLE_AUDIO) -> torch.Tensor:
    """Load a real audio file and compute mel spectrogram matching v5j params."""
    import librosa

    y, sr = librosa.load(audio_path, sr=32000, duration=5.0)
    # Pad to exactly 5s
    target_len = 5 * 32000
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=32000,
        n_fft=2048,
        hop_length=313,
        n_mels=256,
        fmax=None,
    )
    mel_db = librosa.power_to_db(mel, top_db=80.0)
    # Standardize (approximate global stats from v5j)
    mel_norm = (mel_db - (-4.268)) / (4.569)
    mel_tensor = torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 256, T)
    return mel_tensor


class MockEncoder(nn.Module):
    """Minimal encoder that produces realistic feature shapes."""

    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple conv to reduce spatial dims like a real backbone
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=8, stride=8)

    def forward_features(self, x):
        return self.conv(x)  # (B, 1024, F', T')


class OldSEDWrapper(nn.Module):
    """Original SEDWrapper that computes clipwise even when blend_clip=0.
    This is the BUGGY version that causes NaN with AttHead.
    """

    def __init__(self, encoder, sed_head, blend_clip=0.0, blend_frame=1.0):
        super().__init__()
        self.encoder = encoder
        self.sed_head = sed_head
        self.blend_clip = blend_clip
        self.blend_frame = blend_frame

    def forward(self, mel):
        features = self.encoder.forward_features(mel)
        clipwise_logits, framewise_logits = self.sed_head(features)

        if clipwise_logits.dim() == 3:
            n = clipwise_logits.shape[1]
            center = n // 2
            clip_probs = torch.sigmoid(clipwise_logits[:, center, :])
        else:
            clip_probs = torch.sigmoid(clipwise_logits)

        framewise_max_logits = framewise_logits.max(dim=-1).values
        probs = self.blend_clip * clip_probs + self.blend_frame * torch.sigmoid(
            framewise_max_logits
        )
        return probs


class FixedSEDWrapper(nn.Module):
    """Fixed SEDWrapper that skips clipwise when blend_clip=0.
    This avoids NaN from AttHead's torch.logit propagating via 0*NaN=NaN.
    """

    def __init__(self, encoder, sed_head, blend_clip=0.0, blend_frame=1.0):
        super().__init__()
        self.encoder = encoder
        self.sed_head = sed_head
        self.blend_clip = blend_clip
        self.blend_frame = blend_frame

    def forward(self, mel):
        features = self.encoder.forward_features(mel)
        clipwise_logits, framewise_logits = self.sed_head(features)

        framewise_max_logits = framewise_logits.max(dim=-1).values
        frame_probs = torch.sigmoid(framewise_max_logits)

        if self.blend_clip == 0.0:
            return frame_probs

        if clipwise_logits.dim() == 3:
            n = clipwise_logits.shape[1]
            center = n // 2
            clip_probs = torch.sigmoid(clipwise_logits[:, center, :])
        else:
            clip_probs = torch.sigmoid(clipwise_logits)

        if self.blend_frame == 0.0:
            return clip_probs

        return self.blend_clip * clip_probs + self.blend_frame * frame_probs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def real_mel():
    """Load real mel spectrogram from a training audio file."""
    return load_real_mel()


@pytest.fixture
def encoder():
    encoder = MockEncoder(embed_dim=1024)
    encoder.eval()
    return encoder


@pytest.fixture
def atthead():
    head = AttHead(in_channels=1024, num_classes=234, hidden_dim=1024, dropout=0.5)
    head.eval()
    return head


@pytest.fixture
def lsehead():
    head = LSEHead(
        in_channels=1024,
        num_classes=234,
        hidden_dim=1024,
        dropout=0.5,
        freq_pool="mean",
    )
    head.eval()
    return head


class TestAttHeadClipwiseNaN:
    """Test that AttHead's clipwise logits can contain extreme values."""

    def test_atthead_clipwise_is_finite_pytorch(self, encoder, atthead, real_mel):
        """In pure PyTorch, clamp prevents NaN — logit(clamp(x)) is finite."""
        with torch.no_grad():
            features = encoder.forward_features(real_mel)
            clipwise_logits, framewise_logits = atthead(features)
        # PyTorch clamp should keep values finite
        assert torch.isfinite(clipwise_logits).all(), "clipwise has non-finite values in PyTorch"
        assert torch.isfinite(framewise_logits).all(), "framewise has non-finite values in PyTorch"


class TestONNXExport:
    """Test ONNX export and inference for NaN issues."""

    def _export_and_run(self, wrapper, mel_input) -> np.ndarray:
        """Export wrapper to ONNX, run inference, return output."""
        import onnxruntime as ort

        wrapper.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (mel_input,),
                onnx_path,
                opset_version=17,
                input_names=["mel"],
                output_names=["probs"],
                dynamic_axes={"mel": {0: "batch", 3: "time"}, "probs": {0: "batch"}},
            )

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_out = session.run(None, {"mel": mel_input.numpy()})[0]
        Path(onnx_path).unlink()
        return onnx_out

    def test_old_wrapper_atthead_onnx_has_nan(self, encoder, atthead, real_mel):
        """Old wrapper with AttHead + blend 0/1 may produce NaN in ONNX."""
        wrapper = OldSEDWrapper(encoder, atthead, blend_clip=0.0, blend_frame=1.0)
        onnx_out = self._export_and_run(wrapper, real_mel)
        has_nan = np.isnan(onnx_out).any()
        # Document the behavior — NaN is the bug we're fixing
        if has_nan:
            pytest.xfail("Old wrapper produces NaN with AttHead (expected bug)")
        # If no NaN, the bug may be input-dependent or ONNX-version-dependent
        assert np.isfinite(onnx_out).all()

    def test_fixed_wrapper_atthead_onnx_no_nan(self, encoder, atthead, real_mel):
        """Fixed wrapper with AttHead + blend 0/1 must NOT produce NaN."""
        wrapper = FixedSEDWrapper(encoder, atthead, blend_clip=0.0, blend_frame=1.0)
        onnx_out = self._export_and_run(wrapper, real_mel)
        assert np.isfinite(onnx_out).all(), (
            f"Fixed wrapper still produces NaN/inf! "
            f"NaN count: {np.isnan(onnx_out).sum()}, "
            f"Inf count: {np.isinf(onnx_out).sum()}"
        )

    def test_fixed_wrapper_atthead_onnx_reasonable_range(self, encoder, atthead, real_mel):
        """Output probabilities should be in [0, 1]."""
        wrapper = FixedSEDWrapper(encoder, atthead, blend_clip=0.0, blend_frame=1.0)
        onnx_out = self._export_and_run(wrapper, real_mel)
        assert onnx_out.min() >= 0.0, f"Min prob {onnx_out.min()} < 0"
        assert onnx_out.max() <= 1.0, f"Max prob {onnx_out.max()} > 1"

    def test_fixed_wrapper_lsehead_onnx_no_nan(self, encoder, lsehead, real_mel):
        """LSEHead should work fine with both old and new wrapper."""
        wrapper = FixedSEDWrapper(encoder, lsehead, blend_clip=0.0, blend_frame=1.0)
        onnx_out = self._export_and_run(wrapper, real_mel)
        assert np.isfinite(onnx_out).all()

    def test_fixed_wrapper_atthead_blended_onnx_no_nan(self, encoder, atthead, real_mel):
        """AttHead with 0.75/0.25 blend should also work (non-zero blend_clip)."""
        wrapper = FixedSEDWrapper(encoder, atthead, blend_clip=0.75, blend_frame=0.25)
        onnx_out = self._export_and_run(wrapper, real_mel)
        assert np.isfinite(onnx_out).all(), (
            f"Blended wrapper produces NaN/inf with AttHead! "
            f"NaN count: {np.isnan(onnx_out).sum()}"
        )

    def test_pytorch_vs_onnx_match(self, encoder, atthead, real_mel):
        """Fixed wrapper PyTorch output should match ONNX output."""
        wrapper = FixedSEDWrapper(encoder, atthead, blend_clip=0.0, blend_frame=1.0)
        wrapper.eval()

        with torch.no_grad():
            pytorch_out = wrapper(real_mel).numpy()

        onnx_out = self._export_and_run(wrapper, real_mel)
        max_diff = np.max(np.abs(pytorch_out - onnx_out))
        assert max_diff < 1e-4, f"PyTorch vs ONNX max diff: {max_diff}"
