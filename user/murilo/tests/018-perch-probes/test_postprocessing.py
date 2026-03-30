"""Tests for the 6-stage post-processing pipeline.

Cross-ref: ref_yuriy_protossm_v5.py lines 551-694, 2827-2866.
"""

import numpy as np
import pytest

from birdclef.perch.postprocessing import (
    apply_per_class_thresholds,
    delta_shift_smooth,
    file_level_confidence_scale,
    optimize_per_class_thresholds,
    rank_aware_scaling,
    run_postprocessing_pipeline,
    temperature_scale,
)


@pytest.fixture
def dummy_scores():
    """2 files × 12 windows × 10 classes."""
    np.random.seed(42)
    return np.random.randn(24, 10).astype(np.float32)


@pytest.fixture
def dummy_probs():
    """Probabilities in [0, 1]."""
    np.random.seed(42)
    return np.random.rand(24, 10).astype(np.float32)


class TestTemperatureScale:
    def test_shape(self, dummy_scores):
        temps = np.ones(10, dtype=np.float32) * 1.1
        probs = temperature_scale(dummy_scores, temps)
        assert probs.shape == dummy_scores.shape

    def test_output_range(self, dummy_scores):
        temps = np.ones(10, dtype=np.float32)
        probs = temperature_scale(dummy_scores, temps)
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_higher_temp_flattens(self, dummy_scores):
        """Higher temperature → flatter probabilities (closer to 0.5)."""
        probs_t1 = temperature_scale(dummy_scores, np.ones(10))
        probs_t2 = temperature_scale(dummy_scores, np.ones(10) * 2.0)
        # Variance should be lower with higher temperature
        assert probs_t2.var() < probs_t1.var()

    def test_temp_one_equals_sigmoid(self, dummy_scores):
        probs = temperature_scale(dummy_scores, np.ones(10))
        expected = 1.0 / (1.0 + np.exp(-np.clip(dummy_scores, -30, 30)))
        np.testing.assert_allclose(probs, expected, atol=1e-6)


class TestFileLevelConfidence:
    def test_shape(self, dummy_probs):
        scaled = file_level_confidence_scale(dummy_probs, n_windows=12, top_k=2)
        assert scaled.shape == dummy_probs.shape

    def test_scaling_direction(self, dummy_probs):
        """Files with higher top-K mean should have higher scaled values."""
        scaled = file_level_confidence_scale(dummy_probs, n_windows=12, top_k=2)
        # The scaling multiplies by top-K mean, so values can increase
        # but the relative order within a file should be preserved
        view_orig = dummy_probs.reshape(2, 12, 10)
        view_scaled = scaled.reshape(2, 12, 10)
        # For each file, the ranking of windows should be preserved
        for f in range(2):
            for c in range(10):
                orig_rank = np.argsort(view_orig[f, :, c])
                scaled_rank = np.argsort(view_scaled[f, :, c])
                np.testing.assert_array_equal(orig_rank, scaled_rank)


class TestRankAwareScaling:
    def test_shape(self, dummy_probs):
        scaled = rank_aware_scaling(dummy_probs, n_windows=12, power=0.5)
        assert scaled.shape == dummy_probs.shape

    def test_power_zero_identity(self, dummy_probs):
        """power=0 → scale=1 → identity."""
        scaled = rank_aware_scaling(dummy_probs, n_windows=12, power=0.0)
        np.testing.assert_allclose(scaled, dummy_probs, atol=1e-6)


class TestDeltaShiftSmooth:
    def test_shape(self, dummy_probs):
        smoothed = delta_shift_smooth(dummy_probs, n_windows=12, alpha=0.15)
        assert smoothed.shape == dummy_probs.shape

    def test_alpha_zero_identity(self, dummy_probs):
        """alpha=0 → no smoothing."""
        smoothed = delta_shift_smooth(dummy_probs, n_windows=12, alpha=0.0)
        np.testing.assert_allclose(smoothed, dummy_probs, atol=1e-6)

    def test_smoothing_reduces_variance(self, dummy_probs):
        """Smoothing should reduce temporal variance within files."""
        smoothed = delta_shift_smooth(dummy_probs, n_windows=12, alpha=0.5)
        view_orig = dummy_probs.reshape(2, 12, 10)
        view_smooth = smoothed.reshape(2, 12, 10)
        assert view_smooth.var(axis=1).mean() < view_orig.var(axis=1).mean()


class TestPerClassThresholds:
    def test_optimize_basic(self):
        np.random.seed(42)
        scores = np.random.rand(100, 5)
        y_true = (np.random.rand(100, 5) > 0.7).astype(float)
        thresholds, f1s = optimize_per_class_thresholds(scores, y_true)
        assert thresholds.shape == (5,)
        assert f1s.shape == (5,)
        assert all(0.2 <= t <= 0.7 for t in thresholds)

    def test_no_positives_default(self):
        """Classes with no positives should default to 0.5."""
        scores = np.random.rand(100, 3)
        y_true = np.zeros((100, 3))
        thresholds, _ = optimize_per_class_thresholds(scores, y_true)
        np.testing.assert_array_equal(thresholds, [0.5, 0.5, 0.5])

    def test_threshold_half_is_identity(self):
        """With threshold=0.5, sharpening should map 0.5 → 0.5.

        Cross-ref: ref_yuriy_protossm_v5.py lines 688-691.
        """
        scores = np.array([[0.5, 0.3, 0.7]])
        thresholds = np.array([0.5, 0.5, 0.5])
        result = apply_per_class_thresholds(scores, thresholds)
        np.testing.assert_allclose(result, scores, atol=1e-5)

    def test_sharpening_direction(self):
        """Above threshold → pushed higher, below → pushed lower."""
        scores = np.array([[0.8, 0.2]])
        thresholds = np.array([0.5, 0.5])
        result = apply_per_class_thresholds(scores, thresholds)
        assert result[0, 0] > 0.5  # above threshold, pushed higher
        assert result[0, 1] < 0.5  # below threshold, pushed lower

    def test_output_clipped(self):
        scores = np.array([[0.99, 0.01, 0.0, 1.0]])
        thresholds = np.array([0.5, 0.5, 0.5, 0.5])
        result = apply_per_class_thresholds(scores, thresholds)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestFullPipeline:
    def test_output_shape_and_range(self, dummy_scores):
        temps = np.ones(10, dtype=np.float32) * 1.1
        thresholds = np.full(10, 0.5, dtype=np.float32)
        probs = run_postprocessing_pipeline(
            dummy_scores,
            temps,
            thresholds,
            file_level_top_k=2,
            rank_aware_power=0.5,
            delta_shift_alpha=0.15,
        )
        assert probs.shape == dummy_scores.shape
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_pipeline_order_matters(self, dummy_scores):
        """Different pipeline configs should produce different results."""
        temps = np.ones(10, dtype=np.float32) * 1.1
        probs1 = run_postprocessing_pipeline(
            dummy_scores,
            temps,
            file_level_top_k=2,
            rank_aware_power=0.5,
            delta_shift_alpha=0.15,
        )
        probs2 = run_postprocessing_pipeline(
            dummy_scores,
            temps,
            file_level_top_k=0,
            rank_aware_power=0.0,
            delta_shift_alpha=0.0,
        )
        assert not np.allclose(probs1, probs2)
