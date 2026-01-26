"""Test environment setup and basic dependencies."""

import pytest
import torch
import numpy as np


class TestEnvironment:
    """Test basic environment configuration."""

    def test_pytorch_available(self):
        """Test that PyTorch is installed."""
        assert torch.__version__ is not None

    def test_cuda_available(self):
        """Test that CUDA is available."""
        assert torch.cuda.is_available(), "CUDA not available"
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_cuda_version(self):
        """Test CUDA version."""
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            assert cuda_version is not None
            print(f"CUDA version: {cuda_version}")

    def test_gpu_device_name(self):
        """Test GPU device can be queried."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            assert device_name is not None
            print(f"GPU: {device_name}")

    def test_numpy_available(self):
        """Test that NumPy is installed and functional."""
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
