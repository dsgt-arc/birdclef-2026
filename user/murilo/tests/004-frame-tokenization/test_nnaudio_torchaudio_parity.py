"""Parity tests: nnAudio vs torchaudio mel spectrogram output.

Validates that NNAudioMelSpectrogram produces numerically close output
to MelSpectrogramTransform after the filterbank fix (htk=True, norm=None).

Both use identical STFT params (Hann window, power=2.0, same hop/win/nfft),
so the only expected differences are floating-point precision from the
different STFT backends (Conv1d vs torch.stft).
"""

import sys
from pathlib import Path

import pytest
import torch

# Add shared src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdset_benchmark.preprocessing.mel_spectrogram import MelSpectrogramTransform
from birdset_benchmark.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

SAMPLE_RATE = 32000
DURATION = 5.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Tolerance for numerical parity between STFT backends.
# Conv1d (nnAudio) vs torch.stft (torchaudio) can differ at ~1e-4 level
# in linear domain; after log + normalization the difference grows slightly.
ATOL = 0.05
RTOL = 0.02


@pytest.fixture
def nnaudio_mel():
    return NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE)


@pytest.fixture
def torchaudio_mel():
    return MelSpectrogramTransform(sample_rate=SAMPLE_RATE)


def _compute_both(nnaudio_mel, torchaudio_mel, waveform):
    """Compute mel from both backends and return tensors."""
    with torch.no_grad():
        mel_nn = nnaudio_mel(waveform)
        mel_ta = torchaudio_mel(waveform)
    return mel_nn, mel_ta


class TestShapeParity:
    def test_single_shape_matches(self, nnaudio_mel, torchaudio_mel):
        waveform = torch.randn(N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        assert mel_nn.shape == mel_ta.shape

    def test_batch_shape_matches(self, nnaudio_mel, torchaudio_mel):
        waveform = torch.randn(4, N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        assert mel_nn.shape == mel_ta.shape


class TestNumericalParity:
    """Core parity tests: outputs should be nearly identical."""

    def test_random_noise_parity(self, nnaudio_mel, torchaudio_mel):
        """Random noise waveform — broad spectrum exercises all mel bins."""
        torch.manual_seed(42)
        waveform = torch.randn(N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        assert torch.allclose(mel_nn, mel_ta, atol=ATOL, rtol=RTOL), (
            f"Max abs diff: {(mel_nn - mel_ta).abs().max().item():.6f}, "
            f"Mean abs diff: {(mel_nn - mel_ta).abs().mean().item():.6f}"
        )

    def test_sine_440hz_parity(self, nnaudio_mel, torchaudio_mel):
        """Pure 440 Hz sine — tests a single frequency bin region."""
        t = torch.linspace(0, DURATION, N_SAMPLES)
        waveform = 0.5 * torch.sin(2 * torch.pi * 440 * t)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        assert torch.allclose(mel_nn, mel_ta, atol=ATOL, rtol=RTOL), (
            f"Max abs diff: {(mel_nn - mel_ta).abs().max().item():.6f}, "
            f"Mean abs diff: {(mel_nn - mel_ta).abs().mean().item():.6f}"
        )

    def test_sine_4000hz_parity(self, nnaudio_mel, torchaudio_mel):
        """Pure 4000 Hz sine — tests a mid-frequency bin region."""
        t = torch.linspace(0, DURATION, N_SAMPLES)
        waveform = 0.5 * torch.sin(2 * torch.pi * 4000 * t)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        assert torch.allclose(mel_nn, mel_ta, atol=ATOL, rtol=RTOL), (
            f"Max abs diff: {(mel_nn - mel_ta).abs().max().item():.6f}, "
            f"Mean abs diff: {(mel_nn - mel_ta).abs().mean().item():.6f}"
        )

    def test_batch_parity(self, nnaudio_mel, torchaudio_mel):
        """Batch of mixed waveforms."""
        torch.manual_seed(123)
        t = torch.linspace(0, DURATION, N_SAMPLES)
        waveforms = torch.stack(
            [
                torch.randn(N_SAMPLES),  # noise
                0.5 * torch.sin(2 * torch.pi * 440 * t),  # sine
                0.3 * torch.randn(N_SAMPLES) + 0.3 * torch.sin(2 * torch.pi * 2000 * t),
                torch.zeros(N_SAMPLES),  # silence
            ]
        )
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveforms)
        assert torch.allclose(mel_nn, mel_ta, atol=ATOL, rtol=RTOL), (
            f"Max abs diff: {(mel_nn - mel_ta).abs().max().item():.6f}, "
            f"Mean abs diff: {(mel_nn - mel_ta).abs().mean().item():.6f}"
        )

    def test_silence_parity(self, nnaudio_mel, torchaudio_mel):
        """Silent audio — tests log epsilon handling."""
        waveform = torch.zeros(N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        assert torch.allclose(mel_nn, mel_ta, atol=ATOL, rtol=RTOL), (
            f"Max abs diff: {(mel_nn - mel_ta).abs().max().item():.6f}, "
            f"Mean abs diff: {(mel_nn - mel_ta).abs().mean().item():.6f}"
        )


class TestStatisticalParity:
    """Aggregate statistics should match closely."""

    def test_mean_matches(self, nnaudio_mel, torchaudio_mel):
        torch.manual_seed(42)
        waveform = torch.randn(N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        mean_diff = abs(mel_nn.mean().item() - mel_ta.mean().item())
        assert mean_diff < 0.05, f"Mean diff: {mean_diff:.6f}"

    def test_std_matches(self, nnaudio_mel, torchaudio_mel):
        torch.manual_seed(42)
        waveform = torch.randn(N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        std_diff = abs(mel_nn.std().item() - mel_ta.std().item())
        assert std_diff < 0.05, f"Std diff: {std_diff:.6f}"

    def test_value_range_matches(self, nnaudio_mel, torchaudio_mel):
        torch.manual_seed(42)
        waveform = torch.randn(N_SAMPLES)
        mel_nn, mel_ta = _compute_both(nnaudio_mel, torchaudio_mel, waveform)
        min_diff = abs(mel_nn.min().item() - mel_ta.min().item())
        max_diff = abs(mel_nn.max().item() - mel_ta.max().item())
        assert min_diff < 0.1, f"Min diff: {min_diff:.6f}"
        assert max_diff < 0.1, f"Max diff: {max_diff:.6f}"
