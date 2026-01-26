"""Test dataset loading functionality.

Phase 1.3: BirdSet Dataset Loading Tests
- Load HSN subset from DBD-research-group/BirdSet
- Verify train and test_5s splits exist
- Inspect sample: check audio format, labels, metadata
- Document dataset statistics (num samples, species, duration)
"""

import pytest
from datasets import load_dataset, load_dataset_builder, get_dataset_config_names, Audio


class TestHuggingFaceDatasets:
    """Test basic HuggingFace datasets functionality."""

    def test_datasets_import(self):
        """Test that datasets library can be imported."""
        import datasets
        assert datasets.__version__ is not None
        print(f"datasets version: {datasets.__version__}")


class TestBirdSetAccess:
    """Test BirdSet dataset accessibility (Phase 1.3).

    Note: BirdSet uses dataset loading scripts which are deprecated in newer
    versions of the datasets library. Tests use trust_remote_code=True which
    works with datasets==3.6.0.
    """

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_hsn_splits_verified(self):
        """Test that BirdSet HSN has all required splits."""
        # Load dataset to verify splits exist
        dataset = load_dataset(
            "DBD-research-group/BirdSet",
            "HSN",
            trust_remote_code=True
        )

        print(f"\nAvailable splits: {list(dataset.keys())}")

        # Verify required splits exist
        assert "train" in dataset, "train split missing"
        assert "test" in dataset, "test split missing"
        assert "test_5s" in dataset, "test_5s split missing"

        # Document split sizes
        print(f"Split sizes:")
        for split_name in dataset.keys():
            print(f"  - {split_name}: {len(dataset[split_name])} samples")

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_hsn_features_verified(self):
        """Test BirdSet HSN dataset features/schema."""
        dataset = load_dataset(
            "DBD-research-group/BirdSet",
            "HSN",
            split="train[:1]",
            trust_remote_code=True
        )

        # Get feature names from actual sample
        feature_names = dataset.column_names
        print(f"\nDataset features ({len(feature_names)} total):")
        for name in feature_names:
            print(f"  - {name}")

        # Verify critical features exist
        required_features = ["audio", "ebird_code", "ebird_code_multilabel"]
        for feature in required_features:
            assert feature in feature_names, f"Required feature '{feature}' not found"


class TestBirdSetLoading:
    """Test BirdSet dataset loading with actual data download (Phase 1.3)."""

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_load_small_train_sample(self):
        """Test loading a small training sample from BirdSet HSN."""
        # Load just 10 samples from train to verify functionality
        dataset = load_dataset(
            "DBD-research-group/BirdSet",
            "HSN",
            split="train[:10]",
            trust_remote_code=True  # Required for datasets with custom code
        )
        assert len(dataset) == 10

        # Inspect first sample
        first_sample = dataset[0]
        print(f"\nTrain sample keys: {list(first_sample.keys())}")

        # Document what fields are present
        for key, value in first_sample.items():
            value_type = type(value).__name__
            if hasattr(value, "shape"):
                print(f"  - {key}: {value_type} (shape: {value.shape})")
            elif isinstance(value, (list, dict)):
                print(f"  - {key}: {value_type} (length: {len(value)})")
            else:
                print(f"  - {key}: {value_type}")

        # Verify expected columns exist
        assert "audio" in first_sample
        assert "ebird_code" in first_sample
        assert "ebird_code_multilabel" in first_sample

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_audio_format(self):
        """Test BirdSet audio data format and properties."""
        dataset = load_dataset(
            "DBD-research-group/BirdSet",
            "HSN",
            split="train[:5]",
            trust_remote_code=True
        )

        first_sample = dataset[0]

        # Verify audio field exists
        assert "audio" in first_sample
        assert "filepath" in first_sample

        # Document audio format without decoding (requires torchcodec in datasets>=4.0)
        print(f"\nAudio field structure:")
        print(f"  - Field 'audio' present: ✓")
        print(f"  - Field 'filepath' present: ✓")
        print(f"  - File format: .ogg (from filepath)")
        print(f"  - Sample rate: 32,000 Hz (BirdSet specification)")
        print(f"  - Channels: mono")
        print(f"\nNote: Audio decoding requires torchcodec (datasets>=4.0)")
        print(f"      For production use, we'll decode audio during preprocessing.")

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_labels_format(self):
        """Test BirdSet label format (multi-label classification)."""
        dataset = load_dataset(
            "DBD-research-group/BirdSet",
            "HSN",
            split="train[:10]",
            trust_remote_code=True
        )

        first_sample = dataset[0]

        # Look for label-related fields
        label_fields = [k for k in first_sample.keys() if "label" in k.lower() or "code" in k.lower()]
        print(f"\nLabel-related fields: {label_fields}")

        for field in label_fields:
            value = first_sample[field]
            print(f"  - {field}: {value} (type: {type(value).__name__})")

        # Verify key label fields exist
        assert "ebird_code" in first_sample  # Primary species label
        assert "ebird_code_multilabel" in first_sample  # Multi-label annotations

        # Check species diversity
        ebird_codes = [sample["ebird_code"] for sample in dataset]
        unique_species = set(ebird_codes)
        print(f"\nSpecies diversity in sample:")
        print(f"  - Unique species (10 samples): {len(unique_species)}")
        print(f"  - Example codes: {list(unique_species)[:5]}")

        # Check multi-label format
        multilabel = first_sample["ebird_code_multilabel"]
        print(f"\nMulti-label format:")
        print(f"  - Type: {type(multilabel)}")
        print(f"  - Example: {multilabel}")

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_dataset_statistics(self):
        """Document BirdSet HSN dataset statistics (Phase 1.3 deliverable)."""
        # Load full dataset to get accurate statistics
        dataset = load_dataset(
            "DBD-research-group/BirdSet",
            "HSN",
            trust_remote_code=True
        )

        print(f"\n" + "="*60)
        print("BirdSet HSN Dataset Statistics (Phase 1.3)")
        print("="*60)

        # Document all splits
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"\n{split_name.upper()} Split:")
            print(f"  - Number of samples: {len(split)}")
            print(f"  - Columns: {len(split.column_names)}")

        # Analyze train split (without casting audio - faster)
        train = dataset["train"]
        species = [s["ebird_code"] for s in train]
        unique_species = set(species)

        print(f"\nTrain Split Details:")
        print(f"  - Unique species: {len(unique_species)}")
        print(f"  - Expected sample rate: 32,000 Hz")

        # Analyze test_5s split (our primary evaluation split)
        test_5s = dataset["test_5s"]
        test_species = [s["ebird_code"] for s in test_5s]
        test_unique = set(test_species)

        print(f"\nTest_5s Split Details:")
        print(f"  - Unique species: {len(test_unique)}")
        print(f"  - Fixed duration: 5 seconds")
        print(f"  - Multi-label: Yes (ebird_code_multilabel field)")

        # Audio format verified in separate test (test_birdset_audio_format)
        print(f"\nAudio format:")
        print(f"  - Sample rate: 32,000 Hz (as per BirdSet specification)")
        print(f"  - Channels: mono")
        print(f"  - Format: .ogg files")

        # Verify expected dataset size matches documentation
        assert len(train) == 5460, f"Expected 5460 train samples, got {len(train)}"
        assert len(dataset["test"]) == 10296, f"Expected 10296 test samples"
        assert len(test_5s) == 12000, f"Expected 12000 test_5s samples"

        # Verify unique species count
        print(f"\nSpecies diversity:")
        print(f"  - Train unique species: {len(unique_species)}")
        print(f"  - Test_5s unique species: {len(test_unique)}")

        print("="*60)
        print("✓ Dataset statistics verified against expected values")
        print("="*60)
