"""Test mel-spectrogram preprocessing for Experiment 002."""

import torch

from birdset_benchmark.preprocessing.mel_spectrogram import MelSpectrogramTransform


class TestMelSpectrogramTransform:
    """Test MelSpectrogramTransform with Kaldi fbank."""

    def test_transform_instantiation(self):
        """Test that transform can be instantiated with default params."""
        transform = MelSpectrogramTransform(
            sample_rate=32000,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )
        assert transform is not None
        assert transform.sample_rate == 32000
        assert transform.n_mels == 128

    def test_transform_basic_audio(self):
        """Test transform on basic random audio."""
        transform = MelSpectrogramTransform(
            sample_rate=32000,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Create 5-second audio
        waveform = torch.randn(1, 32000 * 5)

        # Transform
        mel_spec = transform(waveform)

        # Check output shape: (1, n_mels, target_frames)
        assert mel_spec.shape == (1, 128, 500), f"Expected (1, 128, 500), got {mel_spec.shape}"

        # Check no NaN or Inf
        assert not torch.isnan(mel_spec).any(), "Output contains NaN"
        assert not torch.isinf(mel_spec).any(), "Output contains Inf"

    def test_transform_normalization(self):
        """Test that normalization is applied correctly."""
        transform = MelSpectrogramTransform(
            sample_rate=32000,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
            norm_mean=-4.268,
            norm_std=4.569,
        )

        # Create audio
        waveform = torch.randn(1, 32000 * 5)
        mel_spec = transform(waveform)

        # Check that output is normalized (should be roughly centered around 0)
        # Note: Won't be exactly 0 mean due to AudioSet normalization constants
        mean = mel_spec.mean().item()
        assert -5 < mean < 5, f"Mean {mean} seems unnormalized"

    def test_transform_different_durations(self):
        """Test transform with different input durations."""
        transform = MelSpectrogramTransform(
            sample_rate=32000,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Test with 3-second audio (shorter than target)
        waveform_short = torch.randn(1, 32000 * 3)
        mel_spec_short = transform(waveform_short)
        assert mel_spec_short.shape == (1, 128, 500), "Should pad to target frames"

        # Test with 8-second audio (longer than target)
        waveform_long = torch.randn(1, 32000 * 8)
        mel_spec_long = transform(waveform_long)
        assert mel_spec_long.shape == (1, 128, 500), "Should crop to target frames"

    def test_transform_batch(self):
        """Test transform on batch of waveforms."""
        transform = MelSpectrogramTransform(
            sample_rate=32000,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Create batch of 4 waveforms
        batch = torch.randn(4, 32000 * 5)

        # Transform each individually (batch processing not supported in transform)
        mel_specs = [transform(waveform.unsqueeze(0)) for waveform in batch]
        mel_specs = torch.cat(mel_specs, dim=0)

        # Check batch output shape
        assert mel_specs.shape == (4, 128, 500), f"Expected (4, 128, 500), got {mel_specs.shape}"

    def test_transform_deterministic(self):
        """Test that transform is deterministic (same input -> same output)."""
        transform = MelSpectrogramTransform(
            sample_rate=32000,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Create audio and transform twice
        waveform = torch.randn(1, 32000 * 5)
        mel_spec1 = transform(waveform)
        mel_spec2 = transform(waveform)

        # Should be identical
        assert torch.allclose(mel_spec1, mel_spec2), "Transform should be deterministic"

    def test_transform_different_n_mels(self):
        """Test transform with different n_mels settings."""
        for n_mels in [64, 128, 256]:
            transform = MelSpectrogramTransform(
                sample_rate=32000,
                n_mels=n_mels,
                frame_shift_ms=10,
                target_frames=500,
            )

            waveform = torch.randn(1, 32000 * 5)
            mel_spec = transform(waveform)

            assert mel_spec.shape == (1, n_mels, 500), f"Expected {n_mels} mel bins"
