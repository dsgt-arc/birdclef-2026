"""Test BirdSetDataModule for Experiment 002."""

import pytest
import torch

from birdclef.lightning.birdset_datamodule import BirdSetDataModule


class TestBirdSetDataModule:
    """Test BirdSetDataModule with on-the-fly mel-spectrogram generation."""

    def test_datamodule_instantiation(self):
        """Test that datamodule can be instantiated."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,  # No workers for quick test
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )
        assert dm is not None
        assert dm.dataset_name == "HSN"
        assert dm.batch_size == 4

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_setup(self):
        """Test datamodule setup (downloads dataset)."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Setup should download and prepare dataset
        dm.setup(stage="fit")

        # Check that datasets are created
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_dataloaders(self):
        """Test that dataloaders can be created."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        dm.setup(stage="fit")

        # Get dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_batch_shapes(self):
        """Test that batches have correct shapes."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        # Get one batch
        batch = next(iter(train_loader))
        mel_specs, labels = batch

        # Check shapes
        assert mel_specs.shape[0] == 4, "Batch size should be 4"
        assert mel_specs.shape[1] == 1, "Should have 1 channel"
        assert mel_specs.shape[2] == 128, "Should have 128 mel bins"
        assert mel_specs.shape[3] == 500, "Should have 500 frames"

        # Labels should be multi-hot (batch_size, num_classes)
        assert labels.shape[0] == 4, "Batch size should be 4"
        assert labels.dim() == 2, "Labels should be 2D (batch, num_classes)"

        # Check dtypes
        assert mel_specs.dtype == torch.float32, "Mel-specs should be float32"
        assert labels.dtype == torch.float32, "Labels should be float32"

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_no_nan_inf(self):
        """Test that batches don't contain NaN or Inf."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        # Check first batch
        batch = next(iter(train_loader))
        mel_specs, labels = batch

        assert not torch.isnan(mel_specs).any(), "Mel-specs contain NaN"
        assert not torch.isinf(mel_specs).any(), "Mel-specs contain Inf"
        assert not torch.isnan(labels).any(), "Labels contain NaN"
        assert not torch.isinf(labels).any(), "Labels contain Inf"

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_multiple_batches(self):
        """Test iteration over multiple batches."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        # Iterate over first 3 batches
        batches = []
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            batches.append(batch)

        assert len(batches) == 3, "Should get 3 batches"

        # Check all batches have correct shapes
        for mel_specs, labels in batches:
            assert mel_specs.shape[1:] == (1, 128, 500), "Incorrect mel-spec shape"
            assert labels.dim() == 2, "Incorrect label shape"

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_with_workers(self):
        """Test datamodule with multiple workers."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=2,  # Use 2 workers
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
            persistent_workers=False,  # Don't persist for test
        )

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        # Should work with workers
        batch = next(iter(train_loader))
        mel_specs, labels = batch

        assert mel_specs.shape[1:] == (1, 128, 500)

    @pytest.mark.slow
    @pytest.mark.download
    def test_datamodule_test_split(self):
        """Test test dataloader."""
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        dm.setup(stage="test")
        test_loader = dm.test_dataloader()

        assert test_loader is not None

        # Get one batch
        batch = next(iter(test_loader))
        mel_specs, labels = batch

        assert mel_specs.shape[1:] == (1, 128, 500)
