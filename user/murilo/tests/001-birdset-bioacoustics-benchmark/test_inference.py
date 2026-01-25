"""Test inference workflow for BirdSet models.

Phase 2A tests: Direct inference with BirdSet-native models (BirdSetConvNeXT, BirdSetEfficientNetB1).
These models use the BMZ .predict(file_paths) API directly - no PyTorch DataLoader needed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from birdset_benchmark.datasets import get_num_classes, load_birdset_dataset
from birdset_benchmark.run_inference import (
    MODELS,
    compute_metrics,
)


class TestMetrics:
    """Test metric computation."""

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        num_classes = 5
        n_samples = 10

        # Perfect predictions
        y_true = np.random.randint(0, 2, size=(n_samples, num_classes)).astype(float)
        y_pred = y_true.copy()

        metrics = compute_metrics(y_true, y_pred, num_classes)

        # Perfect predictions should have high scores
        assert metrics["cmap"] == 1.0
        assert metrics["f1_micro"] == 1.0 or y_true.sum() == 0  # Handle all-zeros case
        print(f"Perfect prediction metrics: {metrics}")

    def test_compute_metrics_random_prediction(self):
        """Test metrics with random predictions."""
        num_classes = 5
        n_samples = 10

        # Random predictions
        y_true = np.random.randint(0, 2, size=(n_samples, num_classes)).astype(float)
        y_pred = np.random.rand(n_samples, num_classes)

        metrics = compute_metrics(y_true, y_pred, num_classes)

        # Check all metrics are computed
        assert "cmap" in metrics
        assert "f1_micro" in metrics
        assert "f1_macro" in metrics
        assert "top1_accuracy" in metrics
        assert "top5_accuracy" in metrics

        # Metrics should be in valid range
        assert 0.0 <= metrics["cmap"] <= 1.0
        assert 0.0 <= metrics["f1_micro"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["top5_accuracy"] <= 1.0

        print(f"Random prediction metrics: {metrics}")

    def test_compute_metrics_zero_predictions(self):
        """Test metrics with all-zero predictions."""
        num_classes = 5
        n_samples = 10

        y_true = np.random.randint(0, 2, size=(n_samples, num_classes)).astype(float)
        y_pred = np.zeros((n_samples, num_classes))

        metrics = compute_metrics(y_true, y_pred, num_classes)

        # Should handle gracefully
        assert metrics["cmap"] >= 0.0
        print(f"Zero prediction metrics: {metrics}")


class TestInferenceBatch:
    """Test batch inference functionality."""

    @pytest.mark.slow
    def test_model_predict_api(self):
        """Test model.predict() API with small batch."""
        # Load a tiny subset
        hf_dataset = load_birdset_dataset(
            dataset_code="HSN",
            split="test_5s",
            subset_size=3,
        )
        num_classes = get_num_classes("HSN")

        # Load model
        model = MODELS["birdset_convnext"]()

        # Get audio paths from first 3 samples
        audio_paths = []
        for i in range(3):
            sample = hf_dataset[i]
            audio_paths.append(sample["audio"]["path"])

        # Run prediction using model's API
        pred_df = model.predict(audio_paths, batch_size=2)

        # Check output is DataFrame
        assert pred_df is not None
        assert pred_df.shape[0] == 3  # 3 samples

        # Get species mapping
        multilabel_feature = hf_dataset.features["ebird_code_multilabel"]
        species_names = multilabel_feature.feature.names
        species_to_idx = {species: idx for idx, species in enumerate(species_names)}

        # Convert to numpy
        y_pred = np.zeros((3, num_classes))
        for species_code in pred_df.columns:
            if species_code in species_to_idx:
                class_idx = species_to_idx[species_code]
                y_pred[:, class_idx] = pred_df[species_code].values

        # Apply sigmoid
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))

        # Check predictions are probabilities (0-1 range)
        assert np.all(y_pred >= 0.0)
        assert np.all(y_pred <= 1.0)

        print(f"Predictions shape: {y_pred.shape}")
        print(f"Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")


class TestInferenceWorkflow:
    """Test end-to-end inference workflow."""

    @pytest.mark.slow
    def test_inference_workflow_small_subset(self):
        """Test full inference workflow on tiny subset."""
        import subprocess
        import sys

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"

            # Run inference script
            cmd = [
                sys.executable,
                "-m",
                "birdset_benchmark.run_inference",
                "--model",
                "birdset_convnext",
                "--dataset",
                "HSN",
                "--subset-size",
                "5",
                "--batch-size",
                "2",
                "--output-dir",
                str(output_dir),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            # Print output for debugging
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert result.returncode == 0, f"Inference failed with return code {result.returncode}"

            # Check outputs exist
            metrics_path = output_dir / "metrics.json"
            predictions_path = output_dir / "predictions.npz"

            assert metrics_path.exists(), f"Metrics file not found at {metrics_path}"
            assert predictions_path.exists(), f"Predictions file not found at {predictions_path}"

            # Load and validate metrics
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Check required metrics
            assert "cmap" in metrics
            assert "f1_micro" in metrics
            assert "f1_macro" in metrics
            assert "top1_accuracy" in metrics
            assert "top5_accuracy" in metrics
            assert "model" in metrics
            assert "dataset" in metrics
            assert "num_samples" in metrics

            # Validate metric values
            assert 0.0 <= metrics["cmap"] <= 1.0
            assert metrics["model"] == "birdset_convnext"
            assert metrics["dataset"] == "HSN"
            assert metrics["num_samples"] == 5

            print("\nInference completed successfully!")
            print(f"cmAP: {metrics['cmap']:.4f}")
            print(f"F1 Micro: {metrics['f1_micro']:.4f}")
            print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")

            # Load and validate predictions
            data = np.load(predictions_path)
            assert "y_pred" in data
            assert "y_true" in data

            y_pred = data["y_pred"]
            y_true = data["y_true"]

            # Check shapes
            assert y_pred.shape[0] == 5  # 5 samples
            assert y_true.shape[0] == 5
            assert y_pred.shape[1] == y_true.shape[1]  # Same number of classes

            print(f"Predictions shape: {y_pred.shape}")
            print(f"Labels shape: {y_true.shape}")

    @pytest.mark.slow
    def test_inference_models_available(self):
        """Test that required models can be loaded."""
        assert "birdset_convnext" in MODELS
        assert "birdset_efficientnet_b1" in MODELS

        # Try loading each model
        for model_name, model_class in MODELS.items():
            print(f"Loading {model_name}...")
            model = model_class()
            assert model is not None
            print(f"  ✓ {model_name} loaded successfully")
