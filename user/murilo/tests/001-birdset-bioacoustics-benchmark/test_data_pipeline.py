"""Test data pipeline for BirdSet benchmark."""

import pytest
import numpy as np
import torch
from birdset_benchmark.data import (
    load_birdset_dataset,
    get_num_classes,
    resample_audio,
    pad_or_crop,
    BirdSetDataset,
    create_dataloader,
    BIRDSET_DATASETS,
)


class TestDataLoading:
    """Test BirdSet dataset loading."""

    def test_birdset_dataset_codes(self):
        """Test that all dataset codes are defined."""
        assert len(BIRDSET_DATASETS) == 8
        assert "HSN" in BIRDSET_DATASETS
        assert "PER" in BIRDSET_DATASETS
        assert "NES" in BIRDSET_DATASETS
        assert "UHH" in BIRDSET_DATASETS
        assert "POW" in BIRDSET_DATASETS
        assert "SSW" in BIRDSET_DATASETS
        assert "SNE" in BIRDSET_DATASETS
        assert "NBP" in BIRDSET_DATASETS

    @pytest.mark.slow
    def test_load_birdset_dataset_small_subset(self):
        """Test loading a small subset of BirdSet HSN dataset."""
        dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=10,
        )
        assert len(dataset) == 10
        assert "audio" in dataset.column_names
        print(f"Dataset columns: {dataset.column_names}")

    def test_invalid_dataset_code(self):
        """Test that invalid dataset code raises error."""
        with pytest.raises(ValueError, match="Unknown dataset code"):
            load_birdset_dataset(dataset_code="INVALID")


class TestAudioPreprocessing:
    """Test audio preprocessing functions."""

    def test_resample_audio_no_change(self):
        """Test resampling with same source and target rate."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = resample_audio(audio, source_sr=16000, target_sr=16000)
        assert np.array_equal(audio, resampled)
        assert resampled.shape == audio.shape

    def test_resample_audio_upsample(self):
        """Test upsampling audio."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = resample_audio(audio, source_sr=16000, target_sr=32000)
        # Upsampling should double the length
        assert resampled.shape[0] == pytest.approx(32000, abs=10)

    def test_resample_audio_downsample(self):
        """Test downsampling audio."""
        audio = np.random.randn(48000).astype(np.float32)
        resampled = resample_audio(audio, source_sr=48000, target_sr=32000)
        # Downsampling should reduce length
        assert resampled.shape[0] == pytest.approx(32000, abs=10)

    def test_pad_or_crop_exact_length(self):
        """Test when audio is already the target length."""
        sr = 32000
        duration = 5.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32)
        result = pad_or_crop(audio, target_duration_sec=duration, sr=sr)
        assert result.shape[0] == int(sr * duration)

    def test_pad_or_crop_too_short(self):
        """Test padding when audio is too short."""
        sr = 32000
        duration = 5.0
        audio = np.random.randn(int(sr * 2.0)).astype(np.float32)  # Only 2s
        result = pad_or_crop(audio, target_duration_sec=duration, sr=sr, mode="pad")
        assert result.shape[0] == int(sr * duration)
        # Check that padding is zeros
        assert np.all(result[len(audio) :] == 0)

    def test_pad_or_crop_too_long(self):
        """Test cropping when audio is too long."""
        sr = 32000
        duration = 5.0
        audio = np.random.randn(int(sr * 10.0)).astype(np.float32)  # 10s
        result = pad_or_crop(audio, target_duration_sec=duration, sr=sr)
        assert result.shape[0] == int(sr * duration)

    def test_pad_or_crop_repeat_mode(self):
        """Test repeat mode for short audio."""
        sr = 32000
        duration = 5.0
        audio = np.random.randn(int(sr * 2.0)).astype(np.float32)  # Only 2s
        result = pad_or_crop(audio, target_duration_sec=duration, sr=sr, mode="repeat")
        assert result.shape[0] == int(sr * duration)
        # Check that audio was repeated (first part matches)
        assert np.array_equal(result[: len(audio)], audio)


class TestBirdSetDataset:
    """Test PyTorch Dataset wrapper."""

    @pytest.mark.slow
    def test_birdset_dataset_creation(self):
        """Test creating a BirdSetDataset instance."""
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=10,
        )
        num_classes = get_num_classes("HSN")
        dataset = BirdSetDataset(
            hf_dataset,
            num_classes=num_classes,
            target_sr=32000,
            target_duration=5.0,
            return_labels=True,
            return_metadata=True,
        )
        assert len(dataset) == 10

    @pytest.mark.slow
    def test_birdset_dataset_getitem(self):
        """Test getting a single item from BirdSetDataset."""
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=5,
        )
        num_classes = get_num_classes("HSN")
        dataset = BirdSetDataset(
            hf_dataset,
            num_classes=num_classes,
            target_sr=32000,
            target_duration=5.0,
            return_labels=True,
            return_metadata=True,
        )

        # Get first item
        sample = dataset[0]

        # Check audio
        assert "audio" in sample
        assert isinstance(sample["audio"], torch.Tensor)
        assert sample["audio"].shape == (32000 * 5,)  # 5s at 32kHz
        assert sample["audio"].dtype == torch.float32

        # Check labels
        assert "labels" in sample
        assert isinstance(sample["labels"], torch.Tensor)
        assert sample["labels"].dtype == torch.float32
        assert sample["labels"].ndim == 1  # Multi-hot vector

        # Check metadata
        assert "metadata" in sample
        assert isinstance(sample["metadata"], dict)
        assert "filename" in sample["metadata"]
        assert "original_sr" in sample["metadata"]

        print(f"Sample audio shape: {sample['audio'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")
        print(f"Sample metadata: {sample['metadata'].keys()}")

    @pytest.mark.slow
    def test_birdset_dataset_audio_resampling(self):
        """Test that audio is properly resampled."""
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=5,
        )

        # Create dataset with different target SR
        dataset = BirdSetDataset(
            hf_dataset,
            num_classes=None,  # Not using labels
            target_sr=48000,  # Different from BirdSet's 32kHz
            target_duration=5.0,
            return_labels=False,
            return_metadata=True,
        )

        sample = dataset[0]
        expected_length = 48000 * 5  # 5s at 48kHz
        assert sample["audio"].shape[0] == expected_length
        print(f"Resampled audio shape: {sample['audio'].shape} (expected: {expected_length})")


class TestDataLoader:
    """Test PyTorch DataLoader."""

    @pytest.mark.slow
    def test_create_dataloader(self):
        """Test creating a DataLoader."""
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=16,
        )
        num_classes = get_num_classes("HSN")
        dataset = BirdSetDataset(
            hf_dataset,
            num_classes=num_classes,
            target_sr=32000,
            target_duration=5.0,
            return_labels=True,
            return_metadata=True,
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
        )

        assert len(dataloader) == 2  # 16 samples / 8 batch size

    @pytest.mark.slow
    def test_dataloader_batching(self):
        """Test that DataLoader correctly batches samples."""
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=16,
        )
        num_classes = get_num_classes("HSN")
        dataset = BirdSetDataset(
            hf_dataset,
            num_classes=num_classes,
            target_sr=32000,
            target_duration=5.0,
            return_labels=True,
            return_metadata=True,
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
        )

        # Get first batch
        batch = next(iter(dataloader))

        # Check audio batch
        assert "audio" in batch
        assert batch["audio"].shape == (8, 32000 * 5)  # [batch, samples]
        assert batch["audio"].dtype == torch.float32

        # Check labels batch
        assert "labels" in batch
        assert batch["labels"].shape[0] == 8  # batch dimension
        assert batch["labels"].dtype == torch.float32

        # Check metadata
        assert "metadata" in batch
        assert len(batch["metadata"]) == 8  # List of metadata dicts

        print(f"Batch audio shape: {batch['audio'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Batch metadata length: {len(batch['metadata'])}")

    @pytest.mark.slow
    def test_dataloader_iteration(self):
        """Test iterating through entire DataLoader."""
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=20,
        )
        num_classes = get_num_classes("HSN")
        dataset = BirdSetDataset(
            hf_dataset,
            num_classes=num_classes,
            target_sr=32000,
            target_duration=5.0,
            return_labels=True,
            return_metadata=False,  # No metadata for speed
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
        )

        num_samples = 0
        for batch in dataloader:
            num_samples += batch["audio"].shape[0]

        assert num_samples == 20
        print(f"Total samples processed: {num_samples}")
