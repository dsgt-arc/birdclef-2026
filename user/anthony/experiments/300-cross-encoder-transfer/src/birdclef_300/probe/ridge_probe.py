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

LAMBDA_GRID = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]


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
        auroc = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
        ap = float(np.mean(fold_aps)) if fold_aps else 0.0
        sweep.append({"lambda": lam, "auroc": auroc, "ap": ap})
        logger.info("  λ=%.3g: AUROC=%.4f AP=%.4f", lam, auroc, ap)
        if auroc > best["macro_auroc"]:
            best = {"best_lambda": lam, "macro_auroc": auroc, "macro_ap": ap}
            best_oof = oof

    # Per-species AUROC at the best λ via that λ's pooled OOF predictions.
    per_species_idx = {j: a for j, (a, _) in _per_column_scores(Y, best_oof).items()}
    if species_list is not None:
        per_species = {species_list[j]: a for j, a in per_species_idx.items()}
    else:
        per_species = {str(j): a for j, a in per_species_idx.items()}

    return {
        "best_lambda": best["best_lambda"],
        "macro_auroc": best["macro_auroc"],
        "macro_ap": best["macro_ap"],
        "n_species_evaluated": len(per_species_idx),
        "n_sites": len(unique_sites),
        "n_windows": int(X.shape[0]),
        "embed_dim": int(X.shape[1]),
        "standardized": standardize,
        "lambda_sweep": sweep,
        "per_species_auroc": per_species,
    }
