"""Verify TuckerMelSpectrogram matches Tucker's reference mel pipeline.

This test guards the Phase D § D.3 fix: training mel must be bit-equivalent
(within fp noise) to Tucker's published `MelSpecTransform` + training-loop
per-sample norm. Without this guard, regressions to a triple-normalized
NNAudio mel like the Phase 1 broken path would silently re-introduce the
training/inference distribution mismatch that collapsed LB to 0.52.

The full TUCKER vs NNAudio-amplitude_to_db side-by-side lives in
`user/murilo/experiments/032-tucker-distilled-sed/scripts/diag_mel_pipeline.py`.
This module just locks in the unit-level guarantee.
"""

import pytest
import torch
import torchaudio

from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram

# Tucker's constants from `bc2026-distilled-sed.py` lines 127-135
SR = 32000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000
TOP_DB = 80


def _tucker_reference(waveform: torch.Tensor) -> torch.Tensor:
    """Verbatim port of Tucker's MelSpecTransform (notebook lines 400-407).

    Returns (B, n_mels, T) — same shape as the original Tucker call.
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=2.0,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
    return db_transform(mel_spec(waveform))


def _tucker_per_sample_norm(mel: torch.Tensor) -> torch.Tensor:
    """Verbatim port of Tucker's training-loop per-sample norm (lines 945-950)."""
    mel = mel.clone()
    B = mel.size(0)
    for i in range(B):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    return mel


@pytest.fixture
def waveform_5s():
    """5s @ 32 kHz of deterministic pseudo-random audio."""
    torch.manual_seed(0)
    return torch.randn(2, SR * 5)  # (B=2, samples=160000)


def test_output_shape(waveform_5s):
    transform = TuckerMelSpectrogram()
    mel = transform(waveform_5s)
    # (B, 1, n_mels, T); T = N // hop + 1 = 160000 // 512 + 1 = 313
    assert mel.shape == (2, 1, N_MELS, 313), mel.shape


def test_dtype_preserved(waveform_5s):
    transform = TuckerMelSpectrogram()
    mel = transform(waveform_5s.float())
    assert mel.dtype == torch.float32


def test_accepts_1d_input():
    """(samples,) → broadcasts to (1, samples) internally."""
    wav = torch.randn(SR * 5)
    transform = TuckerMelSpectrogram()
    mel = transform(wav)
    assert mel.shape == (1, 1, N_MELS, 313)


def test_accepts_3d_input():
    """(B, 1, samples) → squeezes channel for torchaudio MelSpec."""
    wav = torch.randn(2, 1, SR * 5)
    transform = TuckerMelSpectrogram()
    mel = transform(wav)
    assert mel.shape == (2, 1, N_MELS, 313)


def test_finite_output(waveform_5s):
    transform = TuckerMelSpectrogram()
    mel = transform(waveform_5s)
    assert torch.isfinite(mel).all()


def test_matches_tucker_reference_no_norm(waveform_5s):
    """TuckerMelSpectrogram (no per-sample norm) === Tucker's MelSpecTransform.

    Tolerance 1e-5: torchaudio internals + numeric noise.
    """
    ours = TuckerMelSpectrogram()(waveform_5s)  # (B, 1, n_mels, T)
    tucker_ref = _tucker_reference(waveform_5s)  # (B, n_mels, T)
    max_diff = (ours.squeeze(1) - tucker_ref).abs().max().item()
    assert (
        max_diff < 1e-5
    ), f"TuckerMelSpectrogram drifts from Tucker reference: max|d|={max_diff:.4e}"


def test_full_tucker_pipeline_matches(waveform_5s):
    """TuckerMelSpectrogram + per-sample z-score === Tucker's training pipeline.

    This is the load-bearing check: the model trained with `mel.type="tucker"` +
    `per_sample_mel_norm=True` sees the SAME distribution Tucker trains on.
    """
    # Our combined path (mirrors DistilledSEDFinetuneModule._compute_mel)
    mel = TuckerMelSpectrogram()(waveform_5s)  # (B, 1, n_mels, T)
    mean = mel.mean(dim=(1, 2, 3), keepdim=True)
    std = mel.std(dim=(1, 2, 3), keepdim=True)
    ours = (mel - mean) / (std + 1e-6)

    # Tucker's path
    tucker_ref = _tucker_reference(waveform_5s)  # (B, n_mels, T)
    tucker_ref = _tucker_per_sample_norm(tucker_ref)
    tucker_ref = tucker_ref.unsqueeze(1)  # (B, 1, n_mels, T) — match our convention

    max_diff = (ours - tucker_ref).abs().max().item()
    assert max_diff < 1e-5, f"Full pipeline drifts from Tucker: max|d|={max_diff:.4e}"


def test_per_sample_z_score_centered():
    """After the module's per-sample norm, each sample should have mean≈0, std≈1."""
    wav = torch.randn(4, SR * 5)
    mel = TuckerMelSpectrogram()(wav)
    mean = mel.mean(dim=(1, 2, 3), keepdim=True)
    std = mel.std(dim=(1, 2, 3), keepdim=True)
    normed = (mel - mean) / (std + 1e-6)
    per_sample_means = normed.mean(dim=(1, 2, 3))
    per_sample_stds = normed.std(dim=(1, 2, 3))
    assert torch.allclose(per_sample_means, torch.zeros(4), atol=1e-5)
    assert torch.allclose(per_sample_stds, torch.ones(4), atol=1e-3)


def test_top_db_clamps_dynamic_range():
    """AmplitudeToDB(top_db=80) → minimum dB value is `max - 80` per spectrogram."""
    wav = torch.randn(2, SR * 5)
    mel = TuckerMelSpectrogram(top_db=80.0)(wav)
    # Per-sample max - min should be ≤ 80 dB after the clamp
    max_per_sample = mel.amax(dim=(1, 2, 3))
    min_per_sample = mel.amin(dim=(1, 2, 3))
    spread = max_per_sample - min_per_sample
    assert (spread <= 80.0 + 1e-4).all(), spread


def test_params_configurable():
    """All Tucker constants must be settable via __init__."""
    t = TuckerMelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        fmin=50,
        fmax=8000,
        top_db=60,
        power=1.0,
    )
    assert t.sample_rate == 16000
    assert t.n_fft == 1024
    assert t.hop_length == 256
    assert t.n_mels == 128
    assert t.fmin == 50
    assert t.fmax == 8000
    assert t.top_db == 60
    assert t.power == 1.0


def test_to_device_works():
    """Moving the module to a device should move both submodules."""
    t = TuckerMelSpectrogram()
    t.to(torch.device("cpu"))
    # No exception means torchaudio submodules were registered properly
    wav = torch.randn(1, SR * 5)
    mel = t(wav)
    assert mel.device.type == "cpu"


def test_diverges_from_nnaudio_amplitude_to_db(waveform_5s):
    """Regression guard: Tucker pipeline MUST differ from NNAudio's
    `norm_type="amplitude_to_db"` path.

    This is the Phase 1 broken path. If a future refactor accidentally
    routes exp 032 back to NNAudio amplitude_to_db, this test detects it.
    """
    from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

    # Our (correct) Tucker pipeline
    tucker_mel = TuckerMelSpectrogram()(waveform_5s)
    mean = tucker_mel.mean(dim=(1, 2, 3), keepdim=True)
    std = tucker_mel.std(dim=(1, 2, 3), keepdim=True)
    tucker_normed = (tucker_mel - mean) / (std + 1e-6)

    # The broken NNAudio path
    target_frames = int(5.0 * SR / HOP_LENGTH)
    nn_transform = NNAudioMelSpectrogram(
        sample_rate=SR,
        n_mels=N_MELS,
        target_frames=target_frames,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=FMAX,
        norm_type="amplitude_to_db",
        top_db=float(TOP_DB),
    )
    nn_mel = nn_transform(waveform_5s)
    mean = nn_mel.mean(dim=(1, 2, 3), keepdim=True)
    std = nn_mel.std(dim=(1, 2, 3), keepdim=True)
    nn_normed = (nn_mel - mean) / (std + 1e-6)

    # Crop both to the smaller time dim (NNAudio uses target_frames=312;
    # torchaudio gives 313)
    T = min(tucker_normed.shape[-1], nn_normed.shape[-1])
    max_diff = (tucker_normed[..., :T] - nn_normed[..., :T]).abs().max().item()
    # Phase 1 diagnostic recorded max|d| ≈ 3.2 on a real waveform; on
    # synthetic random audio the gap is even larger. Anything below 0.1
    # means the regression guard failed and the two pipelines accidentally
    # converged.
    assert max_diff > 0.1, (
        f"Tucker and NNAudio-amplitude_to_db produced near-identical mels "
        f"(max|d|={max_diff:.4e}). The Phase D fix may have been undone."
    )
