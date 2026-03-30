"""Test-time augmentation, ensemble weight optimization, and score fusion.

Cross-ref: ref_yuriy_protossm_v5.py lines 562-589 (TTA), 2001-2015 (ensemble).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def macro_auc_skip_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute macro AUC, skipping classes with no positive samples."""
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def temporal_shift_tta(
    emb_files: np.ndarray,
    logits_files: np.ndarray,
    model: torch.nn.Module,
    site_ids: np.ndarray,
    hours: np.ndarray,
    shifts: list[int] | None = None,
) -> np.ndarray:
    """TTA by circular-shifting the 12-window embedding sequence.

    For each shift: roll embeddings/logits → forward pass → unroll → average.

    Args:
        emb_files: (n_files, 12, 1536)
        logits_files: (n_files, 12, 234)
        model: ProtoSSMv2 instance
        site_ids: (n_files,) site indices
        hours: (n_files,) hour values
        shifts: list of shifts (default [0, 1, -1])

    Returns:
        averaged_preds: (n_files, 12, 234) averaged logits
    """
    if shifts is None:
        shifts = [0, 1, -1]

    all_preds = []
    model.eval()

    for shift in shifts:
        if shift == 0:
            e = emb_files
            lo = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            lo = np.roll(logits_files, shift, axis=1)

        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(lo, dtype=torch.float32),
                site_ids=torch.tensor(site_ids, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
            pred = out.numpy()

        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def optimize_ensemble_weight(
    oof_proto_flat: np.ndarray,
    oof_mlp_flat: np.ndarray,
    y_true_flat: np.ndarray,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Grid search over blend weights to find optimal ProtoSSM ensemble weight.

    Blended = w * proto + (1-w) * mlp

    Returns:
        best_w: optimal weight for proto
        best_auc: macro AUC at best weight
        results: list of (weight, auc) tuples
    """
    weights = np.arange(0.0, 1.05, 0.05)
    results = []

    for w in weights:
        blended = w * oof_proto_flat + (1.0 - w) * oof_mlp_flat
        try:
            auc = macro_auc_skip_empty(y_true_flat, blended)
        except Exception:
            auc = 0.0
        results.append((float(w), auc))

    best_w, best_auc = max(results, key=lambda x: x[1])
    return best_w, best_auc, results


def fuse_protossm_mlp(
    proto_scores: np.ndarray,
    mlp_scores: np.ndarray,
    weight_proto: float,
) -> np.ndarray:
    """Weighted ensemble of ProtoSSM and MLP probe scores.

    final = weight_proto * proto + (1 - weight_proto) * mlp
    """
    return (weight_proto * proto_scores + (1.0 - weight_proto) * mlp_scores).astype(np.float32)
