"""Phase B unit tests for the class-balanced sampler factory (B.7).

Covers:
- "balanced" mode → w_c = 1/n_c
- "sqrt" mode     → w_c = (n_c/N)^power_value
- unknown mode raises
- empirical: balanced sampler equalizes a skewed class distribution
"""

from collections import Counter

import pytest
import torch

from birdclef.data.xcpretrain_sampler import (
    build_xcpretrain_sampler,
    compute_class_weights,
)


def test_balanced_weights():
    targets = [0] * 90 + [1] * 10
    cw = compute_class_weights(targets, mode="balanced")
    assert cw[0] == pytest.approx(1 / 90)
    assert cw[1] == pytest.approx(1 / 10)


def test_sqrt_weights():
    targets = [0] * 90 + [1] * 10
    cw = compute_class_weights(targets, mode="sqrt", power_value=-0.5)
    n = 100
    assert cw[0] == pytest.approx((90 / n) ** -0.5)
    assert cw[1] == pytest.approx((10 / n) ** -0.5)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="balanced.*sqrt|Unknown mode"):
        compute_class_weights([0, 1], mode="nope")


def test_balanced_total_mass_equal_across_classes():
    # The defining property: under "balanced", each class's TOTAL weight mass
    # (n_c * 1/n_c = 1) is equal, so the sampler draws classes with equal prob.
    targets = [0] * 90 + [1] * 10
    cw = compute_class_weights(targets, mode="balanced")
    mass = {c: sum(cw[t] for t in targets if t == c) for c in (0, 1)}
    assert mass[0] == pytest.approx(mass[1])


def test_balanced_sampler_equalizes_skewed_classes():
    torch.manual_seed(0)
    targets = [0] * 90 + [1] * 10
    sampler = build_xcpretrain_sampler(targets, mode="balanced", with_replacement=True)
    # Aggregate many epochs of draws for a stable empirical frequency.
    drawn = []
    for _ in range(50):
        drawn.extend(targets[i] for i in sampler)
    counts = Counter(drawn)
    frac_minority = counts[1] / len(drawn)
    # Skewed 10% minority should be pulled to ~50% by balanced sampling.
    assert abs(frac_minority - 0.5) < 0.05
