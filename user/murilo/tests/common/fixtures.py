"""Shared pytest fixtures for all experiments.

Common fixtures that can be reused across multiple experiment test suites.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device():
    """Return the CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def temp_cache_dir(tmp_path_factory):
    """Create a temporary directory for caching test data."""
    return tmp_path_factory.mktemp("test_cache")


# Add more shared fixtures as needed
