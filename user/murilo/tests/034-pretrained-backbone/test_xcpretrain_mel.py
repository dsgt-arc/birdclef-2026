"""Phase B unit tests for the Sydorskyi 2025 mel pipeline port.

Covers:
- GeMFreq: p initialized at 3.0 as learnable Parameter
- gem_freq: output shape (B, C, 1, W); exportable vs non-exportable parity
- NormalizeMelSpec: per-sample over (n_mels, time); output range [0, 1]
- ChannelAgnosticAmplitudeToDB: 3D and 4D input both accepted
- TraceableMelspec: output shape matches expected (n_mels, T_frame) for 5s @ 32kHz config
- build_sydorskyi_mel_extractor: full chain end-to-end, finite output, [0, 1] range
"""

import pytest
import torch

from birdclef.preprocessing.xcpretrain_mel import (
    ChannelAgnosticAmplitudeToDB,
    GeMFreq,
    NormalizeMelSpec,
    TraceableMelspec,
    build_sydorskyi_mel_extractor,
    gem_freq,
)


def test_gem_freq_module_init():
    pool = GeMFreq()
    assert isinstance(pool.p, torch.nn.Parameter), "p must be a learnable nn.Parameter"
    assert pool.p.item() == pytest.approx(3.0), f"p default = 3.0, got {pool.p.item()}"
    assert pool.eps == 1e-6


def test_gem_freq_output_shape():
    """GeMFreq pools over freq axis (H), leaves channels + time intact."""
    pool = GeMFreq()
    x = torch.rand(2, 64, 16, 32)  # (B, C, H_freq, W_time)
    out = pool(x)
    assert out.shape == (2, 64, 1, 32), f"got {out.shape}"


def test_gem_freq_exportable_vs_nonexportable_numerical_parity():
    """Exportable branch (Tensor-only) and non-exportable (avg_pool2d) must agree numerically."""
    x = torch.rand(2, 8, 16, 32) + 0.1  # ensure > eps
    y_export = gem_freq(x, p=3.0, exportable=True)
    y_default = gem_freq(x, p=3.0, exportable=False)
    assert torch.allclose(
        y_export, y_default, atol=1e-5
    ), f"Exportable vs default GeM disagree; max diff = {(y_export - y_default).abs().max()}"


def test_normalize_mel_spec_per_sample_standardize():
    """NormalizeMelSpec must standardize per-sample over (n_mels, time) = dims (1, 2)."""
    norm = NormalizeMelSpec(exportable=True)
    # Construct a tensor with KNOWN per-sample mean/std variation
    x = torch.zeros(3, 128, 313)
    x[0] = 5.0  # sample 0: all ones × 5
    x[1] = torch.linspace(-10, 10, 128 * 313).reshape(128, 313)  # sample 1: linear ramp
    x[2] = torch.randn(128, 313) * 2 + 7  # sample 2: random N(7, 4)

    out = norm(x)
    # After standardize + min-max[0, 1], EACH sample's output should span exactly [0, 1]
    for i in range(3):
        if i == 0:
            # Degenerate constant sample → either 0s (non-exportable safeguard) or
            # arbitrary value (exportable) — we just check finite
            assert torch.isfinite(out[i]).all(), f"sample {i} produced non-finite"
        else:
            assert out[i].min() == pytest.approx(
                0.0, abs=1e-5
            ), f"sample {i} min = {out[i].min().item()}"
            assert out[i].max() == pytest.approx(
                1.0, abs=1e-5
            ), f"sample {i} max = {out[i].max().item()}"


def test_normalize_mel_spec_axes_are_per_sample():
    """Confirm standardize is computed over (1, 2), NOT over the whole batch."""
    norm = NormalizeMelSpec(exportable=True)
    # Two samples with intentionally different statistics
    x = torch.cat(
        [
            torch.randn(1, 128, 313) * 1.0 + 0.0,  # mean 0, std 1
            torch.randn(1, 128, 313) * 10.0 + 100.0,  # mean 100, std 10
        ],
        dim=0,
    )
    out = norm(x)
    # If standardize was batch-wide, the two samples would have very different output
    # ranges. With per-sample standardize, both should map to [0, 1].
    for i in range(2):
        assert out[i].min() == pytest.approx(0.0, abs=1e-4)
        assert out[i].max() == pytest.approx(1.0, abs=1e-4)


def test_amplitude_to_db_accepts_3d_and_4d():
    """ChannelAgnosticAmplitudeToDB must accept both 3D (after mel) and 4D (after channel-unsqueeze)."""
    am2db = ChannelAgnosticAmplitudeToDB(top_db=80, amin=1e-10)
    x3 = torch.rand(2, 128, 313) + 1e-3
    x4 = torch.rand(2, 1, 128, 313) + 1e-3
    out3 = am2db(x3)
    out4 = am2db(x4)
    assert out3.shape == (2, 128, 313)
    assert out4.shape == (2, 1, 128, 313)
    # Identical input through both paths must produce identical output
    out_from_3 = am2db(x4.squeeze(1))
    out_from_4 = am2db(x4).squeeze(1)
    assert torch.allclose(out_from_3, out_from_4, atol=1e-6)


def test_amplitude_to_db_top_db_clip():
    """top_db=80 clips dynamic range to ≤80 dB below max."""
    am2db = ChannelAgnosticAmplitudeToDB(top_db=80, amin=1e-10)
    # Power input with huge dynamic range; output dynamic range should be capped at 80 dB
    x = torch.tensor([[[1e-8, 1e-4, 1e0]]])  # 3D (1, 1, 3)
    out = am2db(x).squeeze()
    dyn_range = out.max() - out.min()
    assert dyn_range <= 80.0 + 1e-3, f"dynamic range {dyn_range.item()} exceeds top_db=80"


def test_traceable_melspec_output_shape_5s_32khz():
    """For Sydorskyi config (sr=32k, n_fft=2048, hop=512, n_mels=128), 5s input → (B, 128, 313)."""
    mel = TraceableMelspec(
        sample_rate=32000,
        n_mels=128,
        f_min=20,
        n_fft=2048,
        hop_length=512,
        normalized=True,
    )
    wav = torch.randn(2, 5 * 32000)
    out = mel(wav)
    assert out.shape == (2, 128, 313), f"got {out.shape}"
    assert torch.isfinite(out).all()


def test_traceable_melspec_normalized_flag_scales_output():
    """`normalized=True` divides the STFT by ||hann_window||_2; output should differ from `normalized=False`."""
    kwargs = {"sample_rate": 32000, "n_mels": 128, "f_min": 20, "n_fft": 2048, "hop_length": 512}
    wav = torch.randn(2, 5 * 32000)
    out_norm = TraceableMelspec(normalized=True, **kwargs)(wav)
    out_raw = TraceableMelspec(normalized=False, **kwargs)(wav)
    # They must NOT be equal (would mean the flag is being ignored)
    assert not torch.allclose(
        out_norm, out_raw, atol=1e-6
    ), "normalized=True/False produced identical output"


def test_full_chain_shape_and_range():
    """build_sydorskyi_mel_extractor default chain → final output is in [0, 1], shape (B, n_mels, T_frame)."""
    extractor = build_sydorskyi_mel_extractor(exportable=True)
    wav = torch.randn(2, 5 * 32000)
    out = extractor(wav)
    assert out.shape == (2, 128, 313), f"got {out.shape}"
    assert torch.isfinite(out).all()
    # After NormalizeMelSpec, every per-sample slice spans [0, 1]
    for i in range(2):
        assert out[i].min() == pytest.approx(0.0, abs=1e-4)
        assert out[i].max() == pytest.approx(1.0, abs=1e-4)
