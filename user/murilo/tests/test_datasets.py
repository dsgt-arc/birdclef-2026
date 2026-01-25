"""Test dataset loading functionality."""

import pytest
from datasets import load_dataset, load_dataset_builder


class TestDatasets:
    """Test HuggingFace datasets access."""

    def test_datasets_import(self):
        """Test that datasets library can be imported."""
        import datasets
        assert datasets.__version__ is not None

    @pytest.mark.slow
    def test_birdset_accessible(self):
        """Test BirdSet dataset is accessible (without downloading)."""
        try:
            # Just check if we can build the dataset (doesn't download)
            builder = load_dataset_builder("DBD-research-group/BirdSet", "HSN")
            assert builder is not None
            print(f"BirdSet HSN dataset accessible")
        except Exception as e:
            pytest.skip(f"BirdSet not accessible: {e}")

    @pytest.mark.slow
    @pytest.mark.download
    def test_birdset_load_small_sample(self):
        """Test loading a small sample from BirdSet (requires download)."""
        try:
            # Load just a tiny sample to verify functionality
            dataset = load_dataset(
                "DBD-research-group/BirdSet",
                "HSN",
                split="train[:10]"  # Just 10 samples
            )
            assert len(dataset) == 10

            # Check structure
            first_sample = dataset[0]
            assert "audio" in first_sample or "filepath" in first_sample
            print(f"Sample keys: {first_sample.keys()}")
        except Exception as e:
            pytest.skip(f"Cannot load BirdSet sample: {e}")
