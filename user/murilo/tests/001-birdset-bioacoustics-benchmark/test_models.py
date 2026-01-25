"""Test bioacoustics model zoo integration."""

import pytest
import numpy as np
import tempfile
import soundfile as sf
import os
import bioacoustics_model_zoo as bmz


class TestBioacousticsModelZoo:
    """Test bioacoustics-model-zoo models."""

    def test_bmz_import(self):
        """Test that bioacoustics-model-zoo can be imported."""
        assert bmz.__version__ is not None

    def test_list_models(self):
        """Test listing available models."""
        models = bmz.list_models()
        assert len(models) > 0
        assert "BirdNET" in models
        assert "Perch" in models
        print(f"Available models: {models}")

    def test_birdnet_loading(self):
        """Test BirdNET model can be instantiated."""
        model = bmz.BirdNET()
        assert model is not None

    def test_birdnet_embedding(self):
        """Test BirdNET embedding extraction."""
        model = bmz.BirdNET()

        # Create dummy audio
        sample_rate = 48000  # BirdNET uses 48kHz
        duration = 3.0
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Save to temporary file (BirdNET expects file paths)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, dummy_audio, sample_rate)

            # Extract embedding
            embedding = model.embed(temp_path)

            # Verify shape
            assert embedding.shape[-1] == 1024, f"Expected 1024, got {embedding.shape[-1]}"
            print(f"BirdNET embedding shape: {embedding.shape}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.slow
    def test_perch_loading(self):
        """Test Perch model can be instantiated."""
        model = bmz.Perch()
        assert model is not None

    @pytest.mark.slow
    def test_perch_embedding(self):
        """Test Perch embedding extraction."""
        model = bmz.Perch()

        # Create dummy audio
        sample_rate = 32000  # Perch uses 32kHz
        duration = 5.0  # Perch expects 5s
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, dummy_audio, sample_rate)

            # Extract embedding
            embedding = model.embed(temp_path)

            # Verify shape (Perch outputs 1280-dim embeddings)
            assert embedding.shape[-1] == 1280, f"Expected 1280, got {embedding.shape[-1]}"
            print(f"Perch embedding shape: {embedding.shape}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
