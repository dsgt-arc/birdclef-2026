"""Sydorskyi 2025 class-balanced WeightedRandomSampler — direct port.

Source: vendor/sydorskyi-2025/code_base/train_functions/train_lightning.py:240-263 @ 053594f

Sydorskyi class-balances the training sampler by primary-label frequency. Two
modes (docs/sydorskyi-config-verified.md § Training hyperparameters):

    "balanced"  → w_c = 1 / n_c                  (γ = -1; EffNetV2-S branch,
                                                   selected_ebs.py:214 class_weights_path="balanced")
    "sqrt"      → w_c = (n_c / N) ** power_value  (power_value=-0.5 default;
                                                   NFNet branch, selected_eca.py class_weights_path="sqrt")

Per-sample weight = w_{primary_label(sample)}. The sampler then draws
len(targets) indices per epoch with these weights.

⚠️ Note (verified): the proposal originally had the EffNetV2-S↔NFNet mapping
backwards. EffNetV2-S = balanced (γ=-1), NFNet = sqrt (γ=-0.5).

Sydorskyi uses `replacement=sampler_with_replacement` which defaults to False
(train_lightning.py:200, 263) — WeightedRandomSampler with replacement=False
draws each index at most once per epoch but still biases ordering by weight.

Confidence: VERIFIED (Phase A.2). 2026-05-28.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    targets: Sequence,
    mode: str = "balanced",
    power_value: float = -0.5,
) -> dict:
    """Map each distinct primary-label to its sampling weight.

    Args:
        targets: per-sample primary-label keys (ints or strings), length = dataset size.
        mode: "balanced" (1/n_c) or "sqrt" ((n_c/N)^power_value).
        power_value: exponent for "sqrt" mode (Sydorskyi default -0.5).

    Returns:
        dict {label_key: weight}.
    """
    labels = pd.Series(list(targets))
    vc = labels.value_counts()
    if mode == "sqrt":
        class_weights = (vc / vc.sum()) ** power_value
    elif mode == "balanced":
        class_weights = 1.0 / vc
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'balanced' or 'sqrt'.")
    return class_weights.to_dict()


def build_xcpretrain_sampler(
    targets: Sequence,
    mode: str = "balanced",
    power_value: float = -0.5,
    with_replacement: bool = False,
) -> WeightedRandomSampler:
    """Build a class-balanced WeightedRandomSampler over `targets`.

    Args:
        targets: per-sample primary-label keys (length = dataset size).
        mode: "balanced" (EffNetV2-S) or "sqrt" (NFNet).
        power_value: exponent for "sqrt".
        with_replacement: matches Sydorskyi's `sampler_with_replacement` (default False).

    Returns:
        WeightedRandomSampler drawing len(targets) samples/epoch.
    """
    class_weights = compute_class_weights(targets, mode=mode, power_value=power_value)
    sample_weights = np.array([class_weights[t] for t in targets], dtype=np.float64)
    assert len(sample_weights) == len(targets)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=with_replacement,
    )
