"""BirdSet dataset loading from HuggingFace.

This module provides utilities for loading BirdSet datasets and dataset metadata.

Used by:
- Experiment 001: BirdSet Bioacoustics Benchmark (load_birdset_dataset, get_num_classes)
- Experiment 002: Supervised ViT Baseline (get_num_classes)

Usage:
    from birdset_benchmark.datasets import load_birdset_dataset, get_num_classes, BIRDSET_DATASETS

    # Load a dataset
    dataset = load_birdset_dataset("HSN", split="test_5s")

    # Get number of classes
    num_classes = get_num_classes("HSN")  # Returns 21
"""

import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)


# BirdSet dataset codes mapping
BIRDSET_DATASETS = {
    "HSN": "HSN",  # High Sierra Nevada
    "PER": "PER",  # Amazon Basin, Peru
    "NES": "NES",  # Colombia & Costa Rica
    "UHH": "UHH",  # Hawaiian Islands
    "POW": "POW",  # Powdermill Nature Reserve
    "SSW": "SSW",  # Sapsucker Woods
    "SNE": "SNE",  # Sierra Nevada, Spain
    "NBP": "NBP",  # NIPS4BPlus
}

# Number of species classes per dataset (from BirdSet paper / full datasets)
# These are the true class counts for proper multi-hot encoding
BIRDSET_NUM_CLASSES = {
    "HSN": 21,
    "PER": 132,
    "NES": 43,
    "UHH": 27,
    "POW": 48,
    "SSW": 81,
    "SNE": 52,
    "NBP": 51,
}


def get_num_classes(dataset_code: str) -> int:
    """Get the number of classes for a BirdSet dataset.

    Uses a constant mapping to ensure correct class counts even when using subsets.

    Args:
        dataset_code: Dataset code (HSN, PER, NES, UHH, POW, SSW, SNE, NBP)

    Returns:
        Number of classes for the dataset

    Example:
        >>> num_classes = get_num_classes("HSN")
        >>> print(num_classes)
        21
    """
    if dataset_code not in BIRDSET_NUM_CLASSES:
        raise ValueError(
            f"Unknown dataset code: {dataset_code}. "
            f"Valid codes: {list(BIRDSET_NUM_CLASSES.keys())}"
        )

    num_classes = BIRDSET_NUM_CLASSES[dataset_code]
    logger.info(f"Dataset {dataset_code} has {num_classes} classes")
    return num_classes


def load_birdset_dataset(
    dataset_code: str,
    split: str = "test_5s",
    subset_size: int | None = None,
    cache_dir: str | None = None,
):
    """Load a BirdSet dataset from HuggingFace.

    Args:
        dataset_code: Dataset code (HSN, PER, NES, UHH, POW, SSW, SNE, NBP)
        split: Dataset split ("train", "test_5s", "test_10s")
        subset_size: If specified, only load first N samples (for testing)
        cache_dir: Optional cache directory for datasets

    Returns:
        HuggingFace Dataset object

    Example:
        >>> dataset = load_birdset_dataset("HSN", split="test_5s", subset_size=100)
        >>> print(len(dataset))
        100
    """
    if dataset_code not in BIRDSET_DATASETS:
        raise ValueError(
            f"Unknown dataset code: {dataset_code}. "
            f"Valid codes: {list(BIRDSET_DATASETS.keys())}"
        )

    logger.info(f"Loading BirdSet dataset: {dataset_code}, split: {split}")

    # Load from HuggingFace
    dataset = load_dataset(
        "DBD-research-group/BirdSet",
        dataset_code,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Subset if requested
    if subset_size is not None:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        logger.info(f"Using subset of {len(dataset)} samples")
    else:
        logger.info(f"Loaded {len(dataset)} samples")

    return dataset
