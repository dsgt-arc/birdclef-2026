"""Tests for `SpecAugment` (exp 032 task 0.5).

Validates the Tucker port of cell 10's SpecAugment module:
1. Output shape matches input shape
2. Default params match Tucker exactly: freq_mask_param=10, time_mask_param=10,
   num_freq_masks=1, num_time_masks=2
3. Frequency masking zeroes some frequency rows (across the batch in expectation)
4. Time masking zeroes some time columns (across the batch in expectation)
5. Disabled (num_*_masks=0) → identity transform
6. Output is finite on diverse inputs
7. End-to-end through DistilledSEDFinetuneModule: spec_augment runs during
   training, skipped during eval (matches Tucker's _predict_from_waveforms)

Reference:
- src/birdclef/preprocessing/spec_augment.py
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md (§ "Tucker's mel pipeline")
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 10)
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder
from birdclef.heads.sed_head import AttHeadLogitSpace
from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule
from birdclef.preprocessing.spec_augment import SpecAugment

BATCH = 4
CHANNELS = 1
N_MELS = 256
TIME_FRAMES = 313  # 5s @ 32kHz, hop=512


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def specaug_default():
    """Default SpecAugment = Tucker's exact params."""
    return SpecAugment()


@pytest.fixture
def mel():
    """A clean mel spectrogram with non-zero values everywhere."""
    # Use a constant ramp so we can identify masked regions (zeroed rows/cols)
    # easily without false-positives from random near-zero values.
    return torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES) * 5.0


# ============================================================================
# Defaults match Tucker
# ============================================================================


def test_default_params_match_tucker(specaug_default):
    """Defaults must match Tucker's cell 10 exactly:
    FREQ_MASK_PARAM=10, TIME_MASK_PARAM=10, NUM_FREQ_MASKS=1, NUM_TIME_MASKS=2."""
    assert specaug_default.freq_mask_param == 10
    assert specaug_default.time_mask_param == 10
    assert specaug_default.num_freq_masks == 1
    assert specaug_default.num_time_masks == 2


# ============================================================================
# Output shape + finiteness
# ============================================================================


def test_output_shape_unchanged(specaug_default, mel):
    out = specaug_default(mel)
    assert out.shape == mel.shape
    assert out.dtype == mel.dtype


def test_output_finite(specaug_default, mel):
    out = specaug_default(mel)
    assert torch.isfinite(out).all()


def test_disabled_is_identity(mel):
    """num_freq_masks=0, num_time_masks=0 → identity transform."""
    sa = SpecAugment(num_freq_masks=0, num_time_masks=0)
    out = sa(mel)
    torch.testing.assert_close(out, mel)


# ============================================================================
# Masking actually zeroes regions
# ============================================================================


def test_freq_mask_zeroes_some_rows():
    """With num_freq_masks≥1 and freq_mask_param>0, freq rows get zeroed in
    expectation.

    `torchaudio.transforms.FrequencyMasking` samples a width w ~ Uniform[0, param)
    and a start row from a valid range, then zeroes mel[:, :, start:start+w, :].
    With param=10 and num=1, on average a band of ~5 freq rows is zeroed —
    but a single batch can roll width=0 for all samples (~10%^B probability).
    Run many iterations and check accumulated zeroed rows are non-zero.
    """
    sa = SpecAugment(freq_mask_param=10, time_mask_param=10, num_freq_masks=1, num_time_masks=0)
    total_rows_zeroed = 0
    for _ in range(20):
        mel = torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES) * 5.0
        out = sa(mel)
        for b in range(BATCH):
            row_sums = out[b].sum(dim=(0, 2))  # (n_mels,)
            total_rows_zeroed += int((row_sums == 0).sum().item())
    # Expected: 20 batches × 4 samples × ~5 rows ≈ 400 zeroed rows. Very robust.
    assert (
        total_rows_zeroed > 50
    ), f"freq mask zeroed only {total_rows_zeroed} rows over 80 samples — expected ~400"


def test_time_mask_zeroes_some_columns():
    """With num_time_masks≥1, time cols get zeroed in expectation."""
    sa = SpecAugment(freq_mask_param=10, time_mask_param=10, num_freq_masks=0, num_time_masks=2)
    total_cols_zeroed = 0
    for _ in range(20):
        mel = torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES) * 5.0
        out = sa(mel)
        for b in range(BATCH):
            col_sums = out[b].sum(dim=(0, 1))  # (time_frames,)
            total_cols_zeroed += int((col_sums == 0).sum().item())
    # Expected: 20 × 4 × ~10 (2 masks × 5 width) ≈ 800 zeroed cols. Robust.
    assert (
        total_cols_zeroed > 50
    ), f"time mask zeroed only {total_cols_zeroed} cols over 80 samples — expected ~800"


def test_combined_masking_zeroes_both_axes():
    """1 freq + 2 time masks → some rows AND some cols zeroed."""
    sa = SpecAugment(freq_mask_param=10, time_mask_param=10, num_freq_masks=1, num_time_masks=2)
    # Run many samples so we don't rely on a lucky/unlucky single batch.
    rows_masked = 0
    cols_masked = 0
    for _ in range(20):
        mel = torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES) * 5.0
        out = sa(mel)
        for b in range(BATCH):
            row_sums = out[b].sum(dim=(0, 2))
            col_sums = out[b].sum(dim=(0, 1))
            rows_masked += int((row_sums == 0).sum().item())
            cols_masked += int((col_sums == 0).sum().item())
    # 20 batches × 4 samples × ~5 expected freq-rows ≈ 400 zeroed rows
    # 20 batches × 4 samples × ~10 expected time-cols (2 masks × 5 width) ≈ 800
    assert rows_masked > 0, "no rows masked across 80 samples"
    assert cols_masked > 0, "no cols masked across 80 samples"


def test_freq_mask_width_at_most_param():
    """Mask width is sampled from Uniform[0, freq_mask_param) — never exceeds it.

    With param=20 and num=1, no sample should have > 20 contiguous freq rows zeroed
    by the single mask. (Note: torchaudio's FrequencyMasking may zero a band
    of exactly `param` rows at the upper boundary of the uniform; assert ≤ param.)
    """
    sa = SpecAugment(freq_mask_param=20, time_mask_param=10, num_freq_masks=1, num_time_masks=0)
    mel = torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES) * 5.0
    # Run enough samples to surface large widths
    max_zeroed = 0
    for _ in range(50):
        out = sa(mel.clone())
        for b in range(BATCH):
            row_sums = out[b].sum(dim=(0, 2))
            n_zero = int((row_sums == 0).sum().item())
            max_zeroed = max(max_zeroed, n_zero)
    assert max_zeroed <= 20, f"freq mask zeroed {max_zeroed} rows (> param=20)"


# ============================================================================
# End-to-end: integrates with DistilledSEDFinetuneModule training path
# ============================================================================


class _IdentityMel(nn.Module):
    def forward(self, x):
        # (B, N_samples) → (B, 1, n_mels, time_frames)
        B = x.shape[0]
        return torch.ones(B, 1, N_MELS, TIME_FRAMES) * 5.0


def test_lightning_module_applies_specaugment_in_training():
    """In `DistilledSEDFinetuneModule._compute_mel(x, training=True)`, the
    spec_augment module is called and the output diverges from input.
    With `training=False`, spec_augment is skipped."""
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=234)
    mel = _IdentityMel()
    spec_aug = SpecAugment(
        freq_mask_param=50, time_mask_param=50, num_freq_masks=2, num_time_masks=2
    )  # aggressive masking so the effect is unmistakable
    mod = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=head,
        num_classes=234,
        mel_transform=mel,
        spec_augment=spec_aug,
        per_sample_mel_norm=False,  # so we can detect masking from constant input
    )
    x = torch.randn(BATCH, 160_000)

    # training=True → SpecAugment should fire → some zeros in output mel
    out_train = mod._compute_mel(x, training=True)
    assert (out_train == 0).any().item(), "spec_augment didn't mask in training mode"

    # training=False → SpecAugment skipped → output equals input mel
    out_eval = mod._compute_mel(x, training=False)
    # Without per-sample norm AND without spec_augment, output should equal the
    # mel_transform output exactly.
    expected_eval = mel(x)
    torch.testing.assert_close(out_eval, expected_eval)


def test_lightning_module_handles_none_spec_augment():
    """spec_augment=None (the default) → no masking even at training=True."""
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=234)
    mel = _IdentityMel()
    mod = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=head,
        num_classes=234,
        mel_transform=mel,
        spec_augment=None,
        per_sample_mel_norm=False,
    )
    x = torch.randn(BATCH, 160_000)
    out_train = mod._compute_mel(x, training=True)
    expected = mel(x)
    torch.testing.assert_close(out_train, expected)


# ============================================================================
# Determinism / RNG behavior
# ============================================================================


def test_specaugment_is_stochastic():
    """Different calls produce different masks (stochasticity preserved)."""
    sa = SpecAugment()
    mel = torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES) * 5.0
    out1 = sa(mel.clone())
    out2 = sa(mel.clone())
    # At least one element should differ between two calls (almost surely)
    assert not torch.equal(out1, out2)


def test_specaugment_to_device_works():
    """SpecAugment submodules (FrequencyMasking / TimeMasking) follow .to(device)."""
    sa = SpecAugment()
    # Verify the submodules are registered (so .to() moves them)
    assert isinstance(sa.freq_mask, torch.nn.Module)
    assert isinstance(sa.time_mask, torch.nn.Module)
    # If CUDA is available, exercise .to(device) — otherwise skip (CPU-only env)
    if torch.cuda.is_available():
        sa = sa.to("cuda")
        mel = torch.ones(BATCH, CHANNELS, N_MELS, TIME_FRAMES).cuda()
        out = sa(mel)
        assert out.device.type == "cuda"
