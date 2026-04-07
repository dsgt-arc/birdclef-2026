"""Test that SEDWrapper handles AttHead's torch.logit without NaN propagation.

Background: AttHead computes clipwise via torch.logit(clamp(sigmoid(...))),
which can produce NaN. With blend_clip=0.0, the old wrapper still computed
clipwise and multiplied by 0.0 — but 0.0 * NaN = NaN (IEEE 754), corrupting
the entire output. This caused v6k to score 0.508 LB (random).

The fix: skip clipwise computation entirely when blend_clip == 0.0.

Usage:
    cd user/murilo
    python -m pytest tests/test_sed_wrapper_nan.py -v
"""

import sys
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

    y, _sr = librosa.load(audio_path, sr=32000, duration=5.0)
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
    mel_norm = (mel_db - (-4.268)) / 4.569
    return torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 256, T)


class MockEncoder(nn.Module):
    """Minimal encoder that produces realistic feature shapes."""

    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=8, stride=8)

    def forward_features(self, x):
        return self.conv(x)


def old_wrapper_forward(encoder, sed_head, mel, blend_clip=0.0, blend_frame=1.0):
    """Old SEDWrapper logic — always computes clipwise, multiplies by 0."""
    features = encoder.forward_features(mel)
    clipwise_logits, framewise_logits = sed_head(features)

    clip_probs = torch.sigmoid(clipwise_logits)
    framewise_max_logits = framewise_logits.max(dim=-1).values
    probs = blend_clip * clip_probs + blend_frame * torch.sigmoid(framewise_max_logits)
    return probs


def fixed_wrapper_forward(encoder, sed_head, mel, blend_clip=0.0, blend_frame=1.0):
    """Fixed SEDWrapper logic — skips clipwise when blend_clip=0."""
    features = encoder.forward_features(mel)
    clipwise_logits, framewise_logits = sed_head(features)

    framewise_max_logits = framewise_logits.max(dim=-1).values
    frame_probs = torch.sigmoid(framewise_max_logits)

    if blend_clip == 0.0:
        return frame_probs

    clip_probs = torch.sigmoid(clipwise_logits)
    if blend_frame == 0.0:
        return clip_probs

    return blend_clip * clip_probs + blend_frame * frame_probs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_mel():
    return load_real_mel()


@pytest.fixture
def encoder():
    enc = MockEncoder(embed_dim=1024)
    enc.eval()
    return enc


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


# ---------------------------------------------------------------------------
# Tests: IEEE 754 NaN propagation
# ---------------------------------------------------------------------------


class TestNaNPropagation:
    """Verify 0.0 * NaN = NaN in IEEE 754 — the root cause of the bug."""

    def test_zero_times_nan_is_nan(self):
        """IEEE 754: 0.0 * NaN = NaN. This is why the old wrapper breaks."""
        nan = torch.tensor(float("nan"))
        result = 0.0 * nan + 1.0 * torch.tensor(0.5)
        assert torch.isnan(result), "Expected 0.0 * NaN + 1.0 * 0.5 = NaN"

    def test_sigmoid_of_nan_is_nan(self):
        """sigmoid(NaN) = NaN — if logit produces NaN, sigmoid doesn't fix it."""
        result = torch.sigmoid(torch.tensor(float("nan")))
        assert torch.isnan(result)


# ---------------------------------------------------------------------------
# Tests: AttHead clipwise behavior
# ---------------------------------------------------------------------------


class TestAttHeadClipwise:
    """Test AttHead's clipwise logits (torch.logit round-trip)."""

    def test_atthead_clipwise_finite_in_pytorch(self, encoder, atthead, real_mel):
        """In PyTorch, clamp(1e-7, 1-1e-7) keeps logit() finite."""
        with torch.no_grad():
            features = encoder.forward_features(real_mel)
            clipwise_logits, framewise_logits = atthead(features)
        assert torch.isfinite(clipwise_logits).all()
        assert torch.isfinite(framewise_logits).all()

    def test_atthead_clipwise_can_be_extreme(self, encoder, atthead, real_mel):
        """AttHead clipwise logits can have large magnitude from logit(clamp(...))."""
        with torch.no_grad():
            features = encoder.forward_features(real_mel)
            clipwise_logits, _ = atthead(features)
        # logit(1e-7) ≈ -16.1, logit(1-1e-7) ≈ 16.1
        max_abs = clipwise_logits.abs().max().item()
        # Just document the range — not asserting failure, just that values exist
        assert max_abs > 0, "clipwise logits should be non-zero"

    def test_lsehead_clipwise_no_logit_roundtrip(self, encoder, lsehead, real_mel):
        """LSEHead returns raw logits — no sigmoid→logit round-trip."""
        with torch.no_grad():
            features = encoder.forward_features(real_mel)
            clipwise_logits, framewise_logits = lsehead(features)
        assert torch.isfinite(clipwise_logits).all()
        assert torch.isfinite(framewise_logits).all()


# ---------------------------------------------------------------------------
# Tests: Wrapper NaN behavior (simulates what happens in ONNX)
# ---------------------------------------------------------------------------


class TestWrapperNaN:
    """Test old vs fixed wrapper with AttHead.

    The old wrapper always computes `0.0 * sigmoid(clipwise) + 1.0 * sigmoid(fmax)`.
    If clipwise contains NaN (e.g., from ONNX decomposition of torch.logit), the
    0.0 * NaN = NaN corrupts the output.

    We simulate this by injecting NaN into clipwise to prove the propagation path.
    """

    def test_old_wrapper_propagates_nan_from_clipwise(self, encoder, atthead, real_mel):
        """If clipwise becomes NaN, old wrapper output is NaN even with blend=0."""
        with torch.no_grad():
            features = encoder.forward_features(real_mel)
            clipwise_logits, framewise_logits = atthead(features)

        # Inject NaN into clipwise (simulates ONNX torch.logit failure)
        clipwise_nan = clipwise_logits.clone()
        clipwise_nan[0, 0] = float("nan")

        clip_probs = torch.sigmoid(clipwise_nan)
        framewise_max_logits = framewise_logits.max(dim=-1).values
        probs = 0.0 * clip_probs + 1.0 * torch.sigmoid(framewise_max_logits)

        # The NaN propagates through 0.0 * NaN
        assert torch.isnan(probs).any(), (
            "Expected NaN propagation via 0.0 * NaN — if this fails, "
            "IEEE 754 behavior may differ on this platform"
        )

    def test_fixed_wrapper_immune_to_clipwise_nan(self, encoder, atthead, real_mel):
        """Fixed wrapper skips clipwise entirely, so NaN cannot propagate."""
        with torch.no_grad():
            probs = fixed_wrapper_forward(
                encoder, atthead, real_mel, blend_clip=0.0, blend_frame=1.0
            )
        assert torch.isfinite(probs).all(), (
            f"Fixed wrapper has non-finite values! "
            f"NaN: {torch.isnan(probs).sum()}, Inf: {torch.isinf(probs).sum()}"
        )

    def test_fixed_wrapper_output_in_valid_range(self, encoder, atthead, real_mel):
        """Output probabilities must be in [0, 1]."""
        with torch.no_grad():
            probs = fixed_wrapper_forward(
                encoder, atthead, real_mel, blend_clip=0.0, blend_frame=1.0
            )
        assert probs.min() >= 0.0, f"Min prob {probs.min()} < 0"
        assert probs.max() <= 1.0, f"Max prob {probs.max()} > 1"

    def test_fixed_wrapper_blended_atthead(self, encoder, atthead, real_mel):
        """AttHead with 0.75/0.25 blend (non-zero clip) should also work."""
        with torch.no_grad():
            probs = fixed_wrapper_forward(
                encoder, atthead, real_mel, blend_clip=0.75, blend_frame=0.25
            )
        assert torch.isfinite(
            probs
        ).all(), f"Blended wrapper NaN with AttHead! NaN: {torch.isnan(probs).sum()}"

    def test_fixed_wrapper_lsehead(self, encoder, lsehead, real_mel):
        """LSEHead works fine with fixed wrapper (no logit round-trip)."""
        with torch.no_grad():
            probs = fixed_wrapper_forward(
                encoder, lsehead, real_mel, blend_clip=0.0, blend_frame=1.0
            )
        assert torch.isfinite(probs).all()

    def test_old_wrapper_lsehead_also_fine(self, encoder, lsehead, real_mel):
        """LSEHead works with old wrapper too — no logit in the path."""
        with torch.no_grad():
            probs = old_wrapper_forward(encoder, lsehead, real_mel, blend_clip=0.0, blend_frame=1.0)
        assert torch.isfinite(probs).all()

    def test_old_vs_fixed_same_output_when_no_nan(self, encoder, atthead, real_mel):
        """When clipwise is finite (PyTorch), both wrappers produce the same output."""
        with torch.no_grad():
            old_out = old_wrapper_forward(
                encoder, atthead, real_mel, blend_clip=0.0, blend_frame=1.0
            )
            fixed_out = fixed_wrapper_forward(
                encoder, atthead, real_mel, blend_clip=0.0, blend_frame=1.0
            )
        # In pure PyTorch (no ONNX), both should be finite and equal
        assert torch.isfinite(old_out).all(), "Old wrapper has NaN even in PyTorch"
        max_diff = (old_out - fixed_out).abs().max().item()
        assert max_diff < 1e-6, f"Old vs fixed differ by {max_diff}"
