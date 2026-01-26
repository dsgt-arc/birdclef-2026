"""Test inference workflow for BirdSet models.

Phase 2A tests: Direct inference with BirdSet-native models (BirdSetConvNeXT, BirdSetEfficientNetB1).
These models use the BMZ .predict(file_paths) API directly - no PyTorch DataLoader needed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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


class TestPredictionConversion:
    """Test vectorized prediction conversion."""

    def test_vectorized_conversion_basic(self):
        """Test basic vectorized conversion with dummy data."""

        # Create dummy prediction DataFrame with MultiIndex (like BMZ model output)
        file_paths = [f"/path/to/file_{i}.wav" for i in range(100)]
        species_codes = ["specie_a", "specie_b", "specie_c", "specie_d"]

        # Create MultiIndex DataFrame
        index = pd.MultiIndex.from_arrays([file_paths], names=["file"])
        pred_df = pd.DataFrame(
            np.random.randn(100, 4),  # Logits
            index=index,
            columns=species_codes,
        )

        # Create species mapping
        species_to_idx = {
            "specie_a": 0,
            "specie_b": 1,
            "specie_c": 2,
            "specie_d": 3,
        }
        num_classes = 4
        audio_paths = file_paths  # Same order

        # Vectorized conversion (NEW approach)
        y_pred_new = np.zeros((len(audio_paths), num_classes))

        # Reset MultiIndex to make 'file' a regular index
        if hasattr(pred_df.index, "get_level_values"):
            pred_df_indexed = pred_df.copy()
            pred_df_indexed.index = pred_df.index.get_level_values("file")
        else:
            pred_df_indexed = pred_df

        # Reindex to match audio_paths order
        pred_df_aligned = pred_df_indexed.reindex(audio_paths, fill_value=0.0)

        # Map each species code column to the correct class index
        for species_code in pred_df_aligned.columns:
            if species_code in species_to_idx:
                class_idx = species_to_idx[species_code]
                y_pred_new[:, class_idx] = pred_df_aligned[species_code].values

        # Old approach (REFERENCE for correctness check)
        y_pred_old = np.zeros((len(audio_paths), num_classes))
        for i, audio_path in enumerate(audio_paths):
            matching_rows = pred_df.loc[pred_df.index.get_level_values("file") == audio_path]
            if len(matching_rows) > 0:
                pred_row = matching_rows.iloc[0]
                for species_code in pred_df.columns:
                    if species_code in species_to_idx:
                        class_idx = species_to_idx[species_code]
                        y_pred_old[i, class_idx] = pred_row[species_code]

        # Both approaches should produce identical results
        np.testing.assert_array_almost_equal(y_pred_new, y_pred_old)
        print("✓ Vectorized conversion produces identical results to loop-based approach")

    def test_vectorized_conversion_with_missing_files(self):
        """Test vectorized conversion when some files are missing from predictions."""

        # Predictions for only 3 out of 5 files
        pred_files = ["/path/file_0.wav", "/path/file_2.wav", "/path/file_4.wav"]
        species_codes = ["sp_a", "sp_b"]

        index = pd.MultiIndex.from_arrays([pred_files], names=["file"])
        pred_df = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            index=index,
            columns=species_codes,
        )

        # Request all 5 files
        audio_paths = [f"/path/file_{i}.wav" for i in range(5)]
        species_to_idx = {"sp_a": 0, "sp_b": 1}
        num_classes = 2

        # Vectorized conversion
        y_pred = np.zeros((len(audio_paths), num_classes))

        if hasattr(pred_df.index, "get_level_values"):
            pred_df_indexed = pred_df.copy()
            pred_df_indexed.index = pred_df.index.get_level_values("file")
        else:
            pred_df_indexed = pred_df

        pred_df_aligned = pred_df_indexed.reindex(audio_paths, fill_value=0.0)

        for species_code in pred_df_aligned.columns:
            if species_code in species_to_idx:
                class_idx = species_to_idx[species_code]
                y_pred[:, class_idx] = pred_df_aligned[species_code].values

        # Check: files 0, 2, 4 should have predictions, files 1, 3 should be zeros
        assert y_pred[0, 0] == 1.0 and y_pred[0, 1] == 2.0  # file_0
        assert y_pred[1, 0] == 0.0 and y_pred[1, 1] == 0.0  # file_1 (missing)
        assert y_pred[2, 0] == 3.0 and y_pred[2, 1] == 4.0  # file_2
        assert y_pred[3, 0] == 0.0 and y_pred[3, 1] == 0.0  # file_3 (missing)
        assert y_pred[4, 0] == 5.0 and y_pred[4, 1] == 6.0  # file_4

        print("✓ Vectorized conversion correctly handles missing files")

    def test_vectorized_conversion_performance(self):
        """Test that vectorized conversion is significantly faster for large datasets."""
        import time

        # Simulate SSW-like dataset size
        n_samples = 10000  # Use 10K instead of 205K for reasonable test time
        n_species = 81

        file_paths = [f"/path/to/file_{i}.wav" for i in range(n_samples)]
        species_codes = [f"species_{i:03d}" for i in range(n_species)]

        index = pd.MultiIndex.from_arrays([file_paths], names=["file"])
        pred_df = pd.DataFrame(
            np.random.randn(n_samples, n_species),
            index=index,
            columns=species_codes,
        )

        species_to_idx = {code: i for i, code in enumerate(species_codes)}
        audio_paths = file_paths

        # Time vectorized approach
        start = time.time()
        y_pred_new = np.zeros((len(audio_paths), n_species))

        if hasattr(pred_df.index, "get_level_values"):
            pred_df_indexed = pred_df.copy()
            pred_df_indexed.index = pred_df.index.get_level_values("file")
        else:
            pred_df_indexed = pred_df

        pred_df_aligned = pred_df_indexed.reindex(audio_paths, fill_value=0.0)

        for species_code in pred_df_aligned.columns:
            if species_code in species_to_idx:
                class_idx = species_to_idx[species_code]
                y_pred_new[:, class_idx] = pred_df_aligned[species_code].values

        vectorized_time = time.time() - start

        # Vectorized should complete in reasonable time (< 1 second for 10K samples)
        assert vectorized_time < 1.0, f"Vectorized approach too slow: {vectorized_time:.2f}s"

        print(f"✓ Vectorized conversion completed {n_samples} samples in {vectorized_time:.3f}s")
        print(f"  Estimated time for 205K samples: {vectorized_time * 20.5:.2f}s")

    def test_vectorized_conversion_reordered_paths(self):
        """Test vectorized conversion when audio_paths are in different order than predictions."""

        # Predictions in one order
        pred_files = ["/path/c.wav", "/path/a.wav", "/path/b.wav"]
        species_codes = ["sp_x", "sp_y"]

        index = pd.MultiIndex.from_arrays([pred_files], names=["file"])
        pred_df = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            index=index,
            columns=species_codes,
        )

        # Request in different order
        audio_paths = ["/path/a.wav", "/path/b.wav", "/path/c.wav"]
        species_to_idx = {"sp_x": 0, "sp_y": 1}
        num_classes = 2

        # Vectorized conversion
        y_pred = np.zeros((len(audio_paths), num_classes))

        if hasattr(pred_df.index, "get_level_values"):
            pred_df_indexed = pred_df.copy()
            pred_df_indexed.index = pred_df.index.get_level_values("file")
        else:
            pred_df_indexed = pred_df

        pred_df_aligned = pred_df_indexed.reindex(audio_paths, fill_value=0.0)

        for species_code in pred_df_aligned.columns:
            if species_code in species_to_idx:
                class_idx = species_to_idx[species_code]
                y_pred[:, class_idx] = pred_df_aligned[species_code].values

        # Check correct reordering: a.wav should get [3,4], b.wav [5,6], c.wav [1,2]
        assert y_pred[0, 0] == 3.0 and y_pred[0, 1] == 4.0  # a.wav
        assert y_pred[1, 0] == 5.0 and y_pred[1, 1] == 6.0  # b.wav
        assert y_pred[2, 0] == 1.0 and y_pred[2, 1] == 2.0  # c.wav

        print("✓ Vectorized conversion correctly reorders predictions to match audio_paths")


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
