"""Tests for weaksup.fuse label fusion functions."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# threshold_votes_percentile
# ---------------------------------------------------------------------------


class TestThresholdVotesPercentile:
    def test_basic_shape(self):
        from weaksup.fuse import threshold_votes_percentile

        logits = np.random.randn(100, 5).astype(np.float32)
        votes = threshold_votes_percentile(logits, pos_pct=95.0, neg_pct=50.0)
        assert votes.shape == (100, 5)
        assert votes.dtype == np.int8

    def test_only_valid_values(self):
        from weaksup.fuse import threshold_votes_percentile

        logits = np.random.randn(1000, 3).astype(np.float32)
        votes = threshold_votes_percentile(logits, pos_pct=90.0, neg_pct=50.0)
        assert set(np.unique(votes)).issubset({-1, 0, 1})

    def test_percentile_proportions(self):
        """Top 5% should be +1, bottom 50% should be -1."""
        from weaksup.fuse import threshold_votes_percentile

        rng = np.random.RandomState(42)
        logits = rng.randn(10_000, 1).astype(np.float32)
        votes = threshold_votes_percentile(logits, pos_pct=95.0, neg_pct=50.0)

        pos_frac = (votes == 1).mean()
        neg_frac = (votes == -1).mean()
        # Top 5% -> ~5% positive (using > not >=, so slightly less)
        assert 0.03 <= pos_frac <= 0.07
        # Below 50th percentile -> ~50% negative
        assert 0.45 <= neg_frac <= 0.55

    def test_per_species_independence(self):
        """Each column should be thresholded independently."""
        from weaksup.fuse import threshold_votes_percentile

        rng = np.random.RandomState(42)
        # Column 0: all around 0, Column 1: all around 100
        logits = np.zeros((1000, 2), dtype=np.float32)
        logits[:, 0] = rng.randn(1000).astype(np.float32)
        logits[:, 1] = 100 + rng.randn(1000).astype(np.float32)
        votes = threshold_votes_percentile(logits, pos_pct=95.0, neg_pct=50.0)

        # Both columns should still have +1 and -1 despite different scales
        assert (votes[:, 0] == 1).any()
        assert (votes[:, 1] == 1).any()
        assert (votes[:, 0] == -1).any()
        assert (votes[:, 1] == -1).any()


# ---------------------------------------------------------------------------
# apply_accuracies
# ---------------------------------------------------------------------------


class TestApplyAccuracies:
    def test_uniform_prior_no_votes(self):
        """All abstaining -> output = prevalence."""
        from weaksup.fuse import apply_accuracies

        L = np.zeros((10, 3), dtype=np.int8)
        acc = np.array([0.8, 0.7, 0.9])
        soft = apply_accuracies(L, acc, prevalence=0.5)
        np.testing.assert_allclose(soft, 0.5, atol=1e-5)

    def test_positive_votes_increase_probability(self):
        """All LFs voting +1 should push probability above prevalence."""
        from weaksup.fuse import apply_accuracies

        L = np.ones((5, 3), dtype=np.int8)  # all +1
        acc = np.array([0.8, 0.8, 0.8])
        soft = apply_accuracies(L, acc, prevalence=0.1)
        assert (soft > 0.1).all()

    def test_negative_votes_decrease_probability(self):
        """All LFs voting -1 should push probability below prevalence."""
        from weaksup.fuse import apply_accuracies

        L = -np.ones((5, 3), dtype=np.int8)  # all -1
        acc = np.array([0.8, 0.8, 0.8])
        soft = apply_accuracies(L, acc, prevalence=0.5)
        assert (soft < 0.5).all()

    def test_accuracy_05_is_ignored(self):
        """LFs with accuracy 0.5 should not affect output."""
        from weaksup.fuse import apply_accuracies

        L = np.ones((5, 2), dtype=np.int8)
        acc_with = np.array([0.8, 0.5])
        acc_without = np.array([0.8, 0.5])
        soft = apply_accuracies(L, acc_with, prevalence=0.3)
        # Both should give same result since the 0.5 LF is skipped
        soft2 = apply_accuracies(L[:, :1], acc_without[:1], prevalence=0.3)
        np.testing.assert_allclose(soft, soft2, atol=1e-5)

    def test_prevalence_as_prior(self):
        """Low prevalence should yield lower soft labels even with some positive votes."""
        from weaksup.fuse import apply_accuracies

        L = np.array([[1, 0, 0]], dtype=np.int8)
        acc = np.array([0.7, 0.7, 0.7])
        soft_low = apply_accuracies(L, acc, prevalence=0.01)
        soft_high = apply_accuracies(L, acc, prevalence=0.5)
        assert soft_low[0] < soft_high[0]

    def test_activity_log_odds_suppression(self):
        """Negative activity log-odds should push probability down."""
        from weaksup.fuse import apply_accuracies

        L = np.ones((3, 2), dtype=np.int8)
        acc = np.array([0.8, 0.8])
        soft_no_act = apply_accuracies(L, acc, prevalence=0.5)
        soft_act = apply_accuracies(
            L, acc, prevalence=0.5,
            activity_log_odds=np.full(3, -2.0),
        )
        assert (soft_act < soft_no_act).all()

    def test_output_shape_and_dtype(self):
        from weaksup.fuse import apply_accuracies

        L = np.zeros((20, 4), dtype=np.int8)
        acc = np.array([0.7, 0.7, 0.7, 0.7])
        soft = apply_accuracies(L, acc, prevalence=0.1)
        assert soft.shape == (20,)
        assert soft.dtype == np.float32

    def test_output_in_01_range(self):
        """Output should always be in [0, 1]."""
        from weaksup.fuse import apply_accuracies

        rng = np.random.RandomState(42)
        L = rng.choice([-1, 0, 1], size=(1000, 4)).astype(np.int8)
        acc = np.array([0.9, 0.8, 0.7, 0.6])
        soft = apply_accuracies(
            L, acc, prevalence=0.05,
            activity_log_odds=rng.randn(1000) * 3,
        )
        assert (soft >= 0).all()
        assert (soft <= 1).all()


# ---------------------------------------------------------------------------
# compute_activity_log_odds
# ---------------------------------------------------------------------------


class TestComputeActivityLogOdds:
    def test_no_negative_votes(self):
        """All abstaining -> zero log-odds."""
        from weaksup.fuse import compute_activity_log_odds

        lfs = np.zeros((10, 3), dtype=np.int8)
        result = compute_activity_log_odds(lfs, activity_weight=1.0)
        np.testing.assert_array_equal(result, 0.0)

    def test_all_negative(self):
        """All -1 -> -3 * weight."""
        from weaksup.fuse import compute_activity_log_odds

        lfs = -np.ones((5, 3), dtype=np.int8)
        result = compute_activity_log_odds(lfs, activity_weight=1.0)
        np.testing.assert_array_equal(result, -3.0)

    def test_mixed_votes(self):
        from weaksup.fuse import compute_activity_log_odds

        lfs = np.array([
            [-1, 0, 0],   # 1 negative
            [-1, -1, 0],  # 2 negatives
            [-1, -1, -1], # 3 negatives
            [0, 0, 0],    # 0 negatives
        ], dtype=np.int8)
        result = compute_activity_log_odds(lfs, activity_weight=1.5)
        np.testing.assert_allclose(result, [-1.5, -3.0, -4.5, 0.0])

    def test_weight_scaling(self):
        from weaksup.fuse import compute_activity_log_odds

        lfs = -np.ones((3, 2), dtype=np.int8)
        r1 = compute_activity_log_odds(lfs, activity_weight=1.0)
        r2 = compute_activity_log_odds(lfs, activity_weight=2.0)
        np.testing.assert_allclose(r2, r1 * 2)


# ---------------------------------------------------------------------------
# triplet_method
# ---------------------------------------------------------------------------


class TestTripletMethod:
    def test_returns_array_of_correct_size(self):
        from weaksup.fuse import triplet_method

        rng = np.random.RandomState(42)
        L = rng.choice([-1, 0, 1], size=(500, 3)).astype(np.int8)
        acc = triplet_method(L)
        assert acc.shape == (3,)

    def test_accuracies_in_valid_range(self):
        from weaksup.fuse import triplet_method

        rng = np.random.RandomState(42)
        L = rng.choice([-1, 0, 1], size=(500, 4)).astype(np.int8)
        acc = triplet_method(L)
        assert (acc >= 0.05).all()
        assert (acc <= 0.95).all()

    def test_correlated_lfs_get_high_accuracy(self):
        """LFs that agree with ground truth should get accuracy > 0.5."""
        from weaksup.fuse import triplet_method

        rng = np.random.RandomState(42)
        n = 5000
        # Generate ground truth (balanced for clearer signal)
        y = rng.choice([-1, 1], size=n, p=[0.5, 0.5])
        # 3 LFs that mostly agree with y (80% accuracy)
        L = np.zeros((n, 3), dtype=np.int8)
        for k in range(3):
            flip = rng.rand(n) < 0.2
            L[:, k] = np.where(flip, -y, y)
            # Add some abstentions
            abstain_mask = rng.rand(n) < 0.3
            L[abstain_mask, k] = 0

        acc = triplet_method(L)
        # All should be estimated above 0.5
        assert (acc > 0.5).all(), f"Expected all > 0.5, got {acc}"

    def test_high_accuracy_lf_estimated_higher(self):
        """A more accurate LF should get a higher estimated accuracy."""
        from weaksup.fuse import triplet_method

        rng = np.random.RandomState(42)
        n = 10_000
        y = rng.choice([-1, 1], size=n, p=[0.5, 0.5])

        L = np.zeros((n, 3), dtype=np.int8)
        # LF0: 90% accuracy
        flip0 = rng.rand(n) < 0.1
        L[:, 0] = np.where(flip0, -y, y)
        # LF1: 70% accuracy
        flip1 = rng.rand(n) < 0.3
        L[:, 1] = np.where(flip1, -y, y)
        # LF2: 80% accuracy
        flip2 = rng.rand(n) < 0.2
        L[:, 2] = np.where(flip2, -y, y)

        acc = triplet_method(L)
        assert acc[0] > acc[1], f"LF0 (90%) should be > LF1 (70%), got {acc}"

    def test_all_abstain_gives_05(self):
        """All-abstaining LFs should get accuracy 0.5."""
        from weaksup.fuse import triplet_method

        L = np.zeros((100, 3), dtype=np.int8)
        acc = triplet_method(L)
        np.testing.assert_array_equal(acc, 0.5)

    def test_too_few_non_abstaining_gives_05(self):
        """With < 10 non-abstaining tokens per triplet, should fall back to 0.5."""
        from weaksup.fuse import triplet_method

        L = np.zeros((100, 3), dtype=np.int8)
        # Only 5 non-abstaining tokens (below the min=10 threshold)
        L[:5, :] = 1
        acc = triplet_method(L)
        np.testing.assert_array_equal(acc, 0.5)


# ---------------------------------------------------------------------------
# _fuse_one_species (end-to-end with synthetic data)
# ---------------------------------------------------------------------------


class TestFuseOneSpecies:
    def _make_synthetic_data(self, n=5000, m=4, pos_frac=0.05, seed=42):
        """Create synthetic label matrix with known positive/negative regions."""
        rng = np.random.RandomState(seed)
        n_pos = int(n * pos_frac)

        # Ground truth: first n_pos are positive
        L = np.zeros((n, m), dtype=np.int8)

        # LF0, LF1: species-specific probes (80% acc, 30% abstain)
        for k in range(2):
            votes = np.where(rng.rand(n) < 0.8, 1, -1).astype(np.int8)
            # Negative class gets flipped
            votes[n_pos:] = np.where(
                rng.rand(n - n_pos) < 0.8, -1, 1
            ).astype(np.int8)
            abstain = rng.rand(n) < 0.3
            votes[abstain] = 0
            L[:, k] = votes

        # LF2: weaker probe (60% acc)
        votes = np.where(rng.rand(n) < 0.6, 1, -1).astype(np.int8)
        votes[n_pos:] = np.where(
            rng.rand(n - n_pos) < 0.6, -1, 1
        ).astype(np.int8)
        abstain = rng.rand(n) < 0.5
        votes[abstain] = 0
        L[:, 2] = votes

        # LF3: recording label (always +1 for positive region, abstain elsewhere)
        L[:n_pos, 3] = 1

        return L, n_pos

    def test_end_to_end_shape(self):
        from weaksup.fuse import _fuse_one_species

        L, _ = self._make_synthetic_data(n=2000)
        activity = np.zeros(2000, dtype=np.float64)
        args = (0, "test_species", L, activity, 0.05, 10)
        s_idx, s_code, soft, acc = _fuse_one_species(args)

        assert s_idx == 0
        assert s_code == "test_species"
        assert soft.shape == (2000,)
        assert acc.shape == (4,)

    def test_positive_tokens_get_higher_scores(self):
        """Tokens in the positive region should have higher soft labels on average."""
        from weaksup.fuse import _fuse_one_species

        L, n_pos = self._make_synthetic_data(n=10_000, pos_frac=0.1)
        activity = np.zeros(10_000, dtype=np.float64)
        args = (0, "test_species", L, activity, 0.1, 10)
        _, _, soft, _ = _fuse_one_species(args)

        mean_pos = soft[:n_pos].mean()
        mean_neg = soft[n_pos:].mean()
        assert mean_pos > mean_neg, (
            f"Positive mean {mean_pos:.4f} should be > negative mean {mean_neg:.4f}"
        )

    def test_no_positive_votes_returns_050_accuracy(self):
        """When no LF votes +1, accuracies should be 0.5 (uninformative)."""
        from weaksup.fuse import _fuse_one_species

        # All abstain or negative
        L = np.zeros((100, 4), dtype=np.int8)
        L[:50, 0] = -1
        L[50:, 1] = -1
        activity = np.zeros(100, dtype=np.float64)
        args = (0, "test_species", L, activity, 0.01, 10)
        _, _, soft, acc = _fuse_one_species(args)

        np.testing.assert_array_equal(acc, 0.5)

    def test_activity_suppression_lowers_scores(self):
        """Activity modulation should lower soft labels for suppressed tokens."""
        from weaksup.fuse import _fuse_one_species

        L, n_pos = self._make_synthetic_data(n=2000)
        no_activity = np.zeros(2000, dtype=np.float64)
        suppression = np.full(2000, -2.0)

        args_no = (0, "sp", L, no_activity, 0.05, 10)
        args_sup = (0, "sp", L, suppression, 0.05, 10)

        _, _, soft_no, _ = _fuse_one_species(args_no)
        _, _, soft_sup, _ = _fuse_one_species(args_sup)

        assert soft_sup.mean() < soft_no.mean()
