"""6-stage post-processing pipeline for BirdCLEF predictions.

All stages operate in probability space (after sigmoid), applied in order:
  1. Temperature scaling (logit → sigmoid)
  2. File-level confidence scaling
  3. Rank-aware scaling
  4. Delta shift smoothing
  5. Per-class threshold sharpening

Cross-ref: ref_yuriy_protossm_v5.py lines 551-694, 2827-2866.
"""

from __future__ import annotations

import numpy as np

N_WINDOWS = 12


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def temperature_scale(
    scores: np.ndarray,
    class_temperatures: np.ndarray,
) -> np.ndarray:
    """Stage 1: Per-taxon temperature scaling on logits, then sigmoid.

    Args:
        scores: (N, C) logits
        class_temperatures: (C,) per-class temperature values

    Returns:
        probs: (N, C) probabilities after temperature-scaled sigmoid
    """
    scaled = scores / class_temperatures[None, :]
    return sigmoid(scaled)


def file_level_confidence_scale(
    probs: np.ndarray,
    n_windows: int = N_WINDOWS,
    top_k: int = 2,
) -> np.ndarray:
    """Stage 2: Scale each window by file's top-K mean confidence.

    Cross-ref: ref_yuriy_protossm_v5.py lines 551-559.
    """
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)


def rank_aware_scaling(
    scores: np.ndarray,
    n_windows: int = N_WINDOWS,
    power: float = 0.5,
) -> np.ndarray:
    """Stage 3: Scale each window by (file_max)^power.

    Suppresses predictions in uncertain files, boosts confident files.
    Cross-ref: ref_yuriy_protossm_v5.py lines 595-610.
    """
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows

    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)  # (F, 1, C)
    scale = np.power(file_max, power)
    scaled = view * scale
    return scaled.reshape(N, C)


def delta_shift_smooth(
    scores: np.ndarray,
    n_windows: int = N_WINDOWS,
    alpha: float = 0.15,
) -> np.ndarray:
    """Stage 4: Temporal smoothing across windows.

    new[t] = (1-alpha)*old[t] + 0.5*alpha*(old[t-1] + old[t+1])
    Cross-ref: ref_yuriy_protossm_v5.py lines 613-629.
    """
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows

    view = scores.reshape(n_files, n_windows, C)
    prev_view = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_view = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_view + next_view)
    return smoothed.reshape(N, C)


def optimize_per_class_thresholds(
    oof_scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find optimal per-class F1 threshold from OOF predictions.

    Cross-ref: ref_yuriy_protossm_v5.py lines 632-674.

    Returns:
        best_thresholds: (C,) per-class optimal thresholds
        best_scores: (C,) per-class best F1 scores
    """
    if thresholds is None:
        thresholds = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70], dtype=np.float32)

    n_classes = oof_scores.shape[1]
    best_thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    best_scores = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        y_c = y_true[:, c]
        scores_c = oof_scores[:, c]

        if y_c.sum() == 0:
            continue

        best_f1 = 0.0
        best_t = 0.5

        for t in thresholds:
            pred_c = (scores_c > t).astype(int)
            tp = ((pred_c == 1) & (y_c == 1)).sum()
            fp = ((pred_c == 1) & (y_c == 0)).sum()
            fn = ((pred_c == 0) & (y_c == 1)).sum()

            if tp + fp == 0 or tp + fn == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[c] = best_t
        best_scores[c] = best_f1

    return best_thresholds, best_scores


def apply_per_class_thresholds(
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Stage 5: Sharpen probabilities around per-class thresholds.

    Above threshold: pushed toward 1.0
    Below threshold: pushed toward 0.0
    When thresholds=0.5, this is an identity transform (no-op in submit mode).

    Cross-ref: ref_yuriy_protossm_v5.py lines 677-693.
    """
    N, C = scores.shape
    assert C == len(thresholds)

    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)

    return np.clip(scaled, 0, 1)


def run_postprocessing_pipeline(
    final_scores: np.ndarray,
    class_temperatures: np.ndarray,
    per_class_thresholds: np.ndarray | None = None,
    n_windows: int = N_WINDOWS,
    file_level_top_k: int = 2,
    rank_aware_power: float = 0.5,
    delta_shift_alpha: float = 0.15,
) -> np.ndarray:
    """Run the full 6-stage post-processing pipeline.

    Args:
        final_scores: (N, C) logits
        class_temperatures: (C,) per-class temperature values
        per_class_thresholds: (C,) per-class thresholds (default 0.5 = no-op)
        n_windows: windows per file (12)
        file_level_top_k: K for confidence scaling
        rank_aware_power: power for rank-aware scaling
        delta_shift_alpha: alpha for delta shift smoothing

    Returns:
        probs: (N, C) final probabilities clipped to [0, 1]
    """
    # Stage 1: Temperature scaling (logit space) → probabilities
    probs = temperature_scale(final_scores, class_temperatures)

    # Stage 2: File-level confidence scaling
    if file_level_top_k > 0:
        probs = file_level_confidence_scale(probs, n_windows=n_windows, top_k=file_level_top_k)

    # Stage 3: Rank-aware scaling
    if rank_aware_power > 0:
        probs = rank_aware_scaling(probs, n_windows=n_windows, power=rank_aware_power)

    # Stage 4: Delta shift smoothing
    if delta_shift_alpha > 0:
        probs = delta_shift_smooth(probs, n_windows=n_windows, alpha=delta_shift_alpha)

    # Stage 5: Per-class threshold sharpening
    if per_class_thresholds is not None:
        probs = apply_per_class_thresholds(probs, per_class_thresholds)

    return np.clip(probs, 0, 1)
