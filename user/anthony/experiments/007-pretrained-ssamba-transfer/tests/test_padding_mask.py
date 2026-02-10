"""Tests for patch-aware padding mask computation.

Verifies that _compute_padding_mask produces correct masks for both
tokenizer conditions, matching the Conv2d flatten(2) token ordering.
"""

import pytest
import torch

from src.data import BirdSetDataModule, _compute_padding_mask


class TestPaddingMask128x1:
    """128x1 patches: 1 token per time frame, trivial 1D mask."""

    def test_all_real(self):
        mask = _compute_padding_mask(1000, 1000, (128, 1))
        assert mask.shape == (1000,)
        assert mask.all()

    def test_half_padded(self):
        """5s audio (500 frames) padded to 10s (1000 frames)."""
        mask = _compute_padding_mask(500, 1000, (128, 1))
        assert mask.shape == (1000,)
        assert mask[:500].all()
        assert not mask[500:].any()

    def test_zero_real(self):
        mask = _compute_padding_mask(0, 1000, (128, 1))
        assert not mask.any()

    def test_one_frame(self):
        mask = _compute_padding_mask(1, 1000, (128, 1))
        assert mask[0].item() is True
        assert not mask[1:].any()


class TestPaddingMask16x16:
    """16x16 patches: 2D grid (f_dim=8, t_dim=62), freq-major flatten order.

    Token layout after Conv2d flatten(2):
      token 0  = (f=0, t=0)
      token 1  = (f=0, t=1)
      ...
      token 61 = (f=0, t=61)
      token 62 = (f=1, t=0)
      ...
      token 495 = (f=7, t=61)
    """

    def test_shape(self):
        mask = _compute_padding_mask(1000, 1000, (16, 16))
        assert mask.shape == (496,)  # 8 * 62

    def test_all_real(self):
        mask = _compute_padding_mask(1000, 1000, (16, 16))
        # 1000 // 16 = 62 real t-patches = all t_dim → all True
        assert mask.all()

    def test_half_padded(self):
        """5s audio (500 frames) padded to 10s → 31 real time patches out of 62."""
        mask = _compute_padding_mask(500, 1000, (16, 16))
        real_t = 500 // 16  # 31

        # Check each token
        for i in range(496):
            t_pos = i % 62
            expected = t_pos < real_t
            assert mask[i].item() == expected, f"token {i}: t_pos={t_pos}, expected={expected}"

    def test_freq_rows_consistent(self):
        """All freq rows at the same time position should have the same mask value."""
        mask = _compute_padding_mask(500, 1000, (16, 16))
        t_dim = 62
        f_dim = 8
        for t in range(t_dim):
            values = [mask[f * t_dim + t].item() for f in range(f_dim)]
            assert len(set(values)) == 1, f"Inconsistent mask at t={t}: {values}"

    def test_boundary(self):
        """Exactly 16 real frames → 1 real time-patch."""
        mask = _compute_padding_mask(16, 1000, (16, 16))
        t_dim = 62
        f_dim = 8
        # Only t=0 should be True across all freq rows
        for f in range(f_dim):
            assert mask[f * t_dim + 0].item() is True
            assert mask[f * t_dim + 1].item() is False

    def test_just_under_boundary(self):
        """15 frames → 0 complete time patches → all False."""
        mask = _compute_padding_mask(15, 1000, (16, 16))
        assert not mask.any()

    def test_zero_real(self):
        mask = _compute_padding_mask(0, 1000, (16, 16))
        assert not mask.any()


class TestPaddingMaskCollate:
    """Test that BirdSetDataModule._collate produces correct batch masks."""

    def test_collate_mixed_batch(self):
        """Batch with mixed real/padded samples."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_frames=1000,
            patch_size=(16, 16),
            cache_dir="/nonexistent",  # won't be used
        )

        batch = [
            {"mel": torch.randn(1, 128, 1000), "labels": torch.zeros(21), "real_frames": 1000},
            {"mel": torch.randn(1, 128, 1000), "labels": torch.zeros(21), "real_frames": 500},
            {"mel": torch.randn(1, 128, 1000), "labels": torch.zeros(21), "real_frames": 0},
        ]

        mel_specs, labels, padding_mask = dm._collate(batch)

        assert mel_specs.shape == (3, 1, 128, 1000)
        assert labels.shape == (3, 21)
        assert padding_mask.shape == (3, 496)

        # Sample 0: all real
        assert padding_mask[0].all()
        # Sample 1: half
        assert padding_mask[1].sum() == 8 * 31  # 8 freq rows * 31 time patches
        # Sample 2: none
        assert not padding_mask[2].any()

    def test_collate_128x1(self):
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=2,
            num_workers=0,
            target_frames=1000,
            patch_size=(128, 1),
            cache_dir="/nonexistent",
        )

        batch = [
            {"mel": torch.randn(1, 128, 1000), "labels": torch.zeros(21), "real_frames": 1000},
            {"mel": torch.randn(1, 128, 1000), "labels": torch.zeros(21), "real_frames": 500},
        ]

        mel_specs, labels, padding_mask = dm._collate(batch)
        assert padding_mask.shape == (2, 1000)
        assert padding_mask[0].all()
        assert padding_mask[1, :500].all()
        assert not padding_mask[1, 500:].any()
