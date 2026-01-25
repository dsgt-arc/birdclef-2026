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

    def test_birdset_convnext_loading(self):
        """Test BirdSetConvNeXT model can be instantiated."""
        model = bmz.BirdSetConvNeXT()
        assert model is not None

    def test_birdset_convnext_embedding(self):
        """Test BirdSetConvNeXT embedding extraction."""
        model = bmz.BirdSetConvNeXT()

        # Create dummy audio
        sample_rate = 32000  # BirdSet models use 32kHz
        duration = 5.0  # BirdSet uses 5s windows
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, dummy_audio, sample_rate)

            # Extract embedding
            embedding = model.embed(temp_path)

            # Verify shape (ConvNeXT outputs 1024-dim embeddings)
            assert embedding.shape[-1] == 1024, f"Expected 1024, got {embedding.shape[-1]}"
            print(f"BirdSetConvNeXT embedding shape: {embedding.shape}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_birdset_convnext_predict(self):
        """Test BirdSetConvNeXT can make predictions."""
        model = bmz.BirdSetConvNeXT()

        # Create dummy audio
        sample_rate = 32000
        duration = 5.0
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, dummy_audio, sample_rate)

            # Make predictions
            predictions = model.predict(temp_path)

            # Verify predictions are returned
            assert predictions is not None
            assert hasattr(predictions, 'shape') or isinstance(predictions, (list, dict))
            print(f"BirdSetConvNeXT predictions type: {type(predictions)}")
            if hasattr(predictions, 'shape'):
                print(f"BirdSetConvNeXT predictions shape: {predictions.shape}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_birdset_efficientnet_loading(self):
        """Test BirdSetEfficientNetB1 model can be instantiated."""
        model = bmz.BirdSetEfficientNetB1()
        assert model is not None

    def test_birdset_efficientnet_embedding(self):
        """Test BirdSetEfficientNetB1 embedding extraction."""
        model = bmz.BirdSetEfficientNetB1()

        # Create dummy audio
        sample_rate = 32000  # BirdSet models use 32kHz
        duration = 5.0  # BirdSet uses 5s windows
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, dummy_audio, sample_rate)

            # Extract embedding
            embedding = model.embed(temp_path)

            # Verify shape (EfficientNetB1 outputs 1280-dim embeddings)
            assert embedding.shape[-1] == 1280, f"Expected 1280, got {embedding.shape[-1]}"
            print(f"BirdSetEfficientNetB1 embedding shape: {embedding.shape}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_birdset_efficientnet_predict(self):
        """Test BirdSetEfficientNetB1 can make predictions."""
        model = bmz.BirdSetEfficientNetB1()

        # Create dummy audio
        sample_rate = 32000
        duration = 5.0
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, dummy_audio, sample_rate)

            # Make predictions
            predictions = model.predict(temp_path)

            # Verify predictions are returned
            assert predictions is not None
            assert hasattr(predictions, 'shape') or isinstance(predictions, (list, dict))
            print(f"BirdSetEfficientNetB1 predictions type: {type(predictions)}")
            if hasattr(predictions, 'shape'):
                print(f"BirdSetEfficientNetB1 predictions shape: {predictions.shape}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
