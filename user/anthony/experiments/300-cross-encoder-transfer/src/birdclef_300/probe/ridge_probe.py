"""Ridge linear probe with leave-one-site-out (LOSO) CV.

Ported from exp 106's `ridge_empirical_bayes` soundscape track, reduced to the
ridge-only case the comparison needs (the Hopfield head is intentionally dropped).
The probe is encoder-agnostic: it operates on the (N, D) embedding matrix from any
encoder's unified-cache parquet, so all six encoders sit on one yardstick.

Standardization (per-fold z-score, fit on the train sites) is on by default so the
λ sweep is comparable across encoders with very different embedding scales.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

# Extended to 1e5: with per-fold standardization most encoders rail at the old
# 1000 ceiling, so the optimum sat outside the grid. 1e4/1e5 give it headroom.
LAMBDA_GRID = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def _per_column_scores(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[int, tuple[float, float]]:
    """Per-column (AUROC, AP) for columns with both-class support; degenerate columns skipped."""
    out: dict[int, tuple[float, float]] = {}
    for j in range(Y_true.shape[1]):
        y = Y_true[:, j]
        if y.sum() < 2 or y.sum() >= len(y) - 1:
            continue
        out[j] = (float(roc_auc_score(y, Y_pred[:, j])),
                  float(average_precision_score(y, Y_pred[:, j])))
    return out


def _evaluate(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    scores = _per_column_scores(Y_true, Y_pred)
    aurocs = [a for a, _ in scores.values()]
    aps = [p for _, p in scores.values()]
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "macro_ap": float(np.mean(aps)) if aps else 0.0,
        "n_evaluated": len(aurocs),
    }


def ridge_loso(
    X: np.ndarray,
    Y: np.ndarray,
    sites: np.ndarray,
    species_list: list[str] | None = None,
    standardize: bool = True,
) -> dict:
    """Sweep λ over LOSO folds; report the best macro AUROC/AP and per-species AUROC.

    The primary metric (`macro_auroc`) is the **pooled-OOF** macro ROC-AUC: each
    species' AUROC is computed once over all out-of-fold predictions pooled across
    sites, then averaged across species. This matches the BirdCLEF leaderboard
    metric (one AUROC per class over the whole test set, macro-averaged, skipping
    classes with no positives). λ is selected on this pooled metric.

    The fold-averaged variant (AUROC computed within each site, then averaged over
    sites then species) is reported separately as `fold_avg_auroc` — it measures
    within-site discriminability consistency, a different question, and is NOT how
    the LB scores. (The two can disagree by ~0.06 and even reorder encoders.)

    Args:
        X: (N, D) embeddings. Y: (N, K) binary labels. sites: (N,) site IDs.
        species_list: optional column names for per-species reporting.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    unique_sites = sorted(set(sites.tolist()))

    # Precompute each fold's standardized test block and the Gram products X'X / X'Y,
    # which don't depend on λ — only the +λI diagonal does. The λ sweep then reuses
    # them, turning the expensive Gram from per-(fold, λ) into per-fold.
    folds = []
    for site in unique_sites:
        test = sites == site
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        Xtr, Xte = X[train], X[test]
        if standardize:
            Xtr, Xte = _standardize(Xtr, Xte)
        folds.append({
            "test": test,
            "gram": Xtr.T @ Xtr,
            "xty": Xtr.T @ Y[train],
            "Xte": Xte,
            "Yte": Y[test],
        })

    eye = np.eye(X.shape[1])
    sweep = []
    best = {"macro_auroc": -1.0}
    best_oof: np.ndarray | None = None
    for lam in LAMBDA_GRID:
        oof = np.zeros_like(Y)
        fold_aurocs, fold_aps = [], []
        for f in folds:
            W = scipy.linalg.solve(f["gram"] + lam * eye, f["xty"], assume_a="pos")
            pred = f["Xte"] @ W
            oof[f["test"]] = pred
            m = _evaluate(f["Yte"], pred)
            if m["n_evaluated"] > 0:
                fold_aurocs.append(m["macro_auroc"])
                fold_aps.append(m["macro_ap"])
        # Primary (LB-aligned): per-class AUROC over the pooled OOF, macro-averaged.
        pooled = _evaluate(Y, oof)
        # Secondary: macro-within-fold, then averaged over folds.
        fold_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
        fold_ap = float(np.mean(fold_aps)) if fold_aps else 0.0
        sweep.append({
            "lambda": lam,
            "auroc": pooled["macro_auroc"], "ap": pooled["macro_ap"],
            "fold_auroc": fold_auroc, "fold_ap": fold_ap,
        })
        logger.info("  λ=%.3g: pooled AUROC=%.4f AP=%.4f | fold-avg AUROC=%.4f",
                    lam, pooled["macro_auroc"], pooled["macro_ap"], fold_auroc)
        if pooled["macro_auroc"] > best["macro_auroc"]:
            best = {
                "best_lambda": lam,
                "macro_auroc": pooled["macro_auroc"], "macro_ap": pooled["macro_ap"],
                "fold_avg_auroc": fold_auroc, "fold_avg_ap": fold_ap,
            }
            best_oof = oof

    # Per-species AUROC at the best λ via that λ's pooled OOF predictions.
    per_species_idx = {j: a for j, (a, _) in _per_column_scores(Y, best_oof).items()}
    if species_list is not None:
        per_species = {species_list[j]: a for j, a in per_species_idx.items()}
    else:
        per_species = {str(j): a for j, a in per_species_idx.items()}

    return {
        "best_lambda": best["best_lambda"],
        "macro_auroc": best["macro_auroc"],          # pooled-OOF, LB-aligned (primary)
        "macro_ap": best["macro_ap"],
        "fold_avg_auroc": best["fold_avg_auroc"],     # within-fold then averaged (secondary)
        "fold_avg_ap": best["fold_avg_ap"],
        "n_species_evaluated": len(per_species_idx),
        "n_sites": len(unique_sites),
        "n_windows": int(X.shape[0]),
        "embed_dim": int(X.shape[1]),
        "standardized": standardize,
        "lambda_sweep": sweep,
        "per_species_auroc": per_species,
    }
