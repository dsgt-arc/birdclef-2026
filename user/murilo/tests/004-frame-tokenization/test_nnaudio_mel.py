"""Tests for nnAudio mel spectrogram module.

Validates that NNAudioMelSpectrogram:
1. Produces correct output shapes (single and batched)
2. Runs on GPU when available
3. Produces similar output to the torchaudio-based MelSpectrogramTransform
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


@pytest.fixture
def waveform():
    """Single 5s waveform (simulated audio)."""
    torch.manual_seed(42)
    return torch.randn(N_SAMPLES)


@pytest.fixture
def batch_waveform():
    """Batch of 4 waveforms."""
    torch.manual_seed(42)
    return torch.randn(4, N_SAMPLES)


@pytest.fixture
def nnaudio_mel():
    return NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE)


@pytest.fixture
def torchaudio_mel():
    return MelSpectrogramTransform(sample_rate=SAMPLE_RATE)


class TestOutputShape:
    def test_single_waveform_shape(self, nnaudio_mel, waveform):
        mel = nnaudio_mel(waveform)
        assert mel.shape == (1, 128, 500), f"Expected (1, 128, 500), got {mel.shape}"

    def test_batch_waveform_shape(self, nnaudio_mel, batch_waveform):
        mel = nnaudio_mel(batch_waveform)
        assert mel.shape == (4, 1, 128, 500), f"Expected (4, 1, 128, 500), got {mel.shape}"

    def test_short_audio_padded(self, nnaudio_mel):
        """Audio shorter than target_frames should be zero-padded."""
        short = torch.randn(SAMPLE_RATE)  # 1 second
        mel = nnaudio_mel(short)
        assert mel.shape == (1, 128, 500)

    def test_long_audio_cropped(self, nnaudio_mel):
        """Audio longer than target_frames should be center-cropped."""
        long = torch.randn(SAMPLE_RATE * 10)  # 10 seconds
        mel = nnaudio_mel(long)
        assert mel.shape == (1, 128, 500)


class TestGPU:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_runs_on_gpu(self, waveform):
        device = torch.device("cuda:0")
        mel_layer = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE).to(device)
        mel = mel_layer(waveform.to(device))
        assert mel.device.type == "cuda"
        assert mel.shape == (1, 128, 500)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_batch_on_gpu(self, batch_waveform):
        device = torch.device("cuda:0")
        mel_layer = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE).to(device)
        mel = mel_layer(batch_waveform.to(device))
        assert mel.device.type == "cuda"
        assert mel.shape == (4, 1, 128, 500)


class TestConsistency:
    def test_similar_to_torchaudio(self, nnaudio_mel, torchaudio_mel, waveform):
        """nnAudio and torchaudio should produce nearly identical outputs.

        With matched filterbank params (htk=True, norm=None), the only
        difference is STFT backend (Conv1d vs torch.stft) floating-point
        precision. See test_nnaudio_torchaudio_parity.py for thorough tests.
        """
        mel_nn = nnaudio_mel(waveform)
        mel_ta = torchaudio_mel(waveform)

        assert (
            mel_nn.shape == mel_ta.shape
        ), f"Shape mismatch: nnAudio {mel_nn.shape} vs torchaudio {mel_ta.shape}"

        mean_diff = abs(mel_nn.mean().item() - mel_ta.mean().item())
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"

    def test_no_nans(self, nnaudio_mel, waveform):
        mel = nnaudio_mel(waveform)
        assert not torch.isnan(mel).any(), "Output contains NaN values"

    def test_no_infs(self, nnaudio_mel, waveform):
        mel = nnaudio_mel(waveform)
        assert not torch.isinf(mel).any(), "Output contains Inf values"

    def test_silent_audio(self, nnaudio_mel):
        """Silent audio should produce finite output (no log(0) issues)."""
        silent = torch.zeros(N_SAMPLES)
        mel = nnaudio_mel(silent)
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()
