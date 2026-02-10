"""Tests for the cached mel data pipeline.

Validates dataset loading, collation, and integration with the model.
Requires the pre-computed mel cache to exist.
"""

import pytest
import torch

from src.data import BirdSetDataModule, CachedMelDataset
from tests.conftest import requires_cache, CACHE_DIR


@requires_cache
class TestCachedMelDataset:
    """Test loading pre-computed mel spectrograms."""

    def test_train_dataset_loads(self):
        ds = CachedMelDataset(CACHE_DIR / "train")
        assert len(ds) > 0
        print(f"Train dataset: {len(ds)} samples")

    def test_val_dataset_loads(self):
        ds = CachedMelDataset(CACHE_DIR / "val")
        assert len(ds) > 0
        print(f"Val dataset: {len(ds)} samples")

    def test_sample_shape(self):
        ds = CachedMelDataset(CACHE_DIR / "train")
        sample = ds[0]

        assert "mel" in sample
        assert "labels" in sample
        assert "real_frames" in sample

        assert sample["mel"].shape == (1, 128, 1000), f"Got {sample['mel'].shape}"
        assert sample["labels"].shape == (21,), f"Got {sample['labels'].shape}"
        assert 0 <= sample["real_frames"] <= 1000

    def test_labels_valid(self):
        """Labels should be binary, not all zero for at least some samples."""
        ds = CachedMelDataset(CACHE_DIR / "train")
        any_positive = False
        any_negative = False
        for i in range(min(100, len(ds))):
            labels = ds[i]["labels"]
            assert labels.min() >= 0.0 and labels.max() <= 1.0
            if labels.sum() > 0:
                any_positive = True
            else:
                any_negative = True
        assert any_positive, "No positive labels found in first 100 samples"
        # Negatives are expected from non-event windows
        print(f"Positive samples found: {any_positive}, Negative samples: {any_negative}")

    def test_real_frames_distribution(self):
        """Training samples should mostly have real_frames=1000 (full 10s windows)."""
        ds = CachedMelDataset(CACHE_DIR / "train")
        full_count = 0
        n = min(100, len(ds))
        for i in range(n):
            if ds[i]["real_frames"] >= 990:  # allow small rounding
                full_count += 1
        pct = 100 * full_count / n
        print(f"Train samples with ~full frames: {pct:.0f}%")
        assert pct > 80, f"Expected most train samples to be full 10s, got {pct:.0f}%"

    def test_val_real_frames(self):
        """Val samples are 5s padded to 10s, so real_frames should be ~500."""
        ds = CachedMelDataset(CACHE_DIR / "val")
        sample = ds[0]
        rf = sample["real_frames"]
        # 5s @ 100Hz = 500 frames, allow some tolerance
        assert 400 <= rf <= 600, f"Expected ~500 real_frames for 5s val clip, got {rf}"


@requires_cache
class TestDataModuleIntegration:
    """Test BirdSetDataModule with real cached data."""

    @pytest.fixture
    def dm_16x16(self):
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=8,
            num_workers=0,
            target_frames=1000,
            patch_size=(16, 16),
            cache_dir=str(CACHE_DIR),
        )
        dm.setup("fit")
        return dm

    @pytest.fixture
    def dm_128x1(self):
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=8,
            num_workers=0,
            target_frames=1000,
            patch_size=(128, 1),
            cache_dir=str(CACHE_DIR),
        )
        dm.setup("fit")
        return dm

    def test_train_batch_16x16(self, dm_16x16):
        loader = dm_16x16.train_dataloader()
        mel, labels, mask = next(iter(loader))

        assert mel.shape == (8, 1, 128, 1000)
        assert labels.shape == (8, 21)
        assert mask.shape == (8, 496)
        assert mask.dtype == torch.bool

    def test_val_batch_16x16(self, dm_16x16):
        loader = dm_16x16.val_dataloader()
        mel, labels, mask = next(iter(loader))

        assert mel.shape[1:] == (1, 128, 1000)
        assert mask.shape[1] == 496
        # Val samples should have partial masks (5s padded to 10s)
        assert not mask.all(), "Val mask should have some False values (padding)"

    def test_train_batch_128x1(self, dm_128x1):
        loader = dm_128x1.train_dataloader()
        mel, labels, mask = next(iter(loader))

        assert mel.shape == (8, 1, 128, 1000)
        assert mask.shape == (8, 1000)

    def test_val_batch_128x1(self, dm_128x1):
        loader = dm_128x1.val_dataloader()
        mel, labels, mask = next(iter(loader))

        assert mask.shape[1] == 1000
        assert not mask.all(), "Val mask should have padding"
