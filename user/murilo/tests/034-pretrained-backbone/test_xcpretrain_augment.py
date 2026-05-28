"""Phase B unit tests for the augmentation ports (B.8.1 spec_augment + B.8.2 random_filtering).

Covers:
- CustomFreqMasking / CustomTimeMasking: shape preserved; non-negative input energy
  never increases under masking; p=0 is a no-op; p=1 actually masks; inplace=False
  does not mutate the input.
- RandomFiltering: waveform roundtrip is finite and ~same length; normalize_wave
  caps peak amplitude at ~1; is_wave=False applies gain to a spectrogram directly.
"""

import torch

from birdclef.preprocessing.random_filtering import RandomFiltering
from birdclef.preprocessing.xcpretrain_spec_augment import (
    CustomFreqMasking,
    CustomTimeMasking,
)


def test_masking_shape_and_no_energy_increase():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 128, 313)  # non-negative
    for Mask in (CustomFreqMasking, CustomTimeMasking):
        m = Mask(mask_max_length=15, mask_max_masks=3, p=1.0, inplace=False)
        out = m(x)
        assert out.shape == x.shape
        # masking only zeros entries → total energy cannot increase
        assert out.sum().item() <= x.sum().item() + 1e-4


def test_masking_p_zero_is_noop():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 128, 313)
    m = CustomFreqMasking(mask_max_length=15, mask_max_masks=3, p=0.0, inplace=False)
    torch.testing.assert_close(m(x), x)


def test_masking_p_one_changes_tensor():
    # Over several trials with p=1.0 at least one must change the tensor.
    changed = False
    for seed in range(10):
        torch.manual_seed(seed)
        x = torch.rand(2, 1, 128, 313)
        m = CustomFreqMasking(mask_max_length=20, mask_max_masks=3, p=1.0, inplace=False)
        if not torch.allclose(m(x), x):
            changed = True
            break
    assert changed


def test_masking_inplace_false_does_not_mutate_input():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 128, 313)
    x_ref = x.clone()
    m = CustomTimeMasking(mask_max_length=20, mask_max_masks=3, p=1.0, inplace=False)
    _ = m(x)
    torch.testing.assert_close(x, x_ref)


def test_random_filtering_wave_roundtrip_finite():
    torch.manual_seed(0)
    wave = torch.randn(2, 160_000)
    rf = RandomFiltering(n_bands=4, min_db=-20, is_wave=True, normalize_wave=True)
    out = rf(wave)
    assert torch.isfinite(out).all()
    # istft length should be within one hop of the input length
    assert abs(out.shape[-1] - wave.shape[-1]) <= rf.hop_length
    # peak-normalized → |amp| <= ~1
    assert out.abs().max().item() <= 1.0 + 1e-3


def test_random_filtering_spec_path_applies_gain():
    torch.manual_seed(0)
    spec = torch.rand(2, 513, 100)  # (B, bins, time), non-negative
    rf = RandomFiltering(n_bands=4, min_db=-20, is_wave=False)
    out = rf(spec)
    assert out.shape == spec.shape
    # gains are 10^(dB/20) with dB in [min_db, 0] → in (0, 1], so output <= input
    assert (out <= spec + 1e-5).all()
