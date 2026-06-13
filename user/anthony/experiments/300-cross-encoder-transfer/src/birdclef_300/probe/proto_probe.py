"""Prototypical LOSO probe — the MAE-appropriate readout.

Linear ridge probing under-reads masked-SSL (MAE-style) features: Bird-MAE (Rauch et
al. 2025) reports linear probe 13.3 mAP vs prototypical 49.0 vs fine-tune 55.3 on MAE
features. So for the exp 301 native-objective DAPT we report the frozen→adapted Δ under
*both* ridge and this prototypical readout, with the SAME readout on frozen and
adapted (apples-to-apples).

Two strategies (ported from exp 016 `zero_training_baseline.py`):
  * Strategy A (clip-level cosine) — prototype per species = mean of L2-normed pooled
    embeddings of its positive windows; score = cosine(pooled, prototype).
  * Strategy B (frame-level max) — same prototypes, but score = max over patches of
    cosine(patch, prototype). This is the readout that closes the Bird-MAE gap; it
    needs the per-patch cache (`patches.parquet`, written by `extract --patch`).

Both run LOSO with prototypes built from **train sites only** per fold (same no-leak
protocol as ridge; the DAPT pool is unlabeled regardless). The metric is the identical
pooled-OOF macro ROC-AUC as the ridge probe (reusing its `_evaluate`), so the two
readouts are directly comparable.
"""

from __future__ import annotations

import logging

import numpy as np

from birdclef_300.probe.ridge_probe import _evaluate, _per_column_scores

logger = logging.getLogger(__name__)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def proto_loso(
    X: np.ndarray,
    Y: np.ndarray,
    sites: np.ndarray,
    patches: list[np.ndarray] | None = None,
    species_list: list[str] | None = None,
) -> dict:
    """Prototypical LOSO readout, pooled-OOF macro AUROC (LB-aligned, like ridge).

    Args:
        X: (N, D) pooled embeddings. Y: (N, K) binary labels. sites: (N,) site IDs.
        patches: optional list of (P, D) per-window patch grids → enables Strategy B.
        species_list: optional column names for per-species reporting.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    K = Y.shape[1]
    Xn = _l2(X)
    unique_sites = sorted(set(sites.tolist()))

    oof_a = np.zeros_like(Y)
    oof_b = np.zeros_like(Y) if patches is not None else None

    for site in unique_sites:
        test = sites == site
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        protos = np.zeros((K, X.shape[1]))
        for k in range(K):
            m = (Y[:, k] == 1) & train
            if m.any():
                protos[k] = Xn[m].mean(axis=0)
        protos = _l2(protos)
        oof_a[test] = Xn[test] @ protos.T            # Strategy A: clip-level cosine
        if patches is not None:
            for i in np.where(test)[0]:
                pp = _l2(np.asarray(patches[i], dtype=np.float64))   # (P, D)
                oof_b[i] = (pp @ protos.T).max(axis=0)               # Strategy B: frame max

    eval_a = _evaluate(Y, oof_a)
    per_a_idx = {j: a for j, (a, _) in _per_column_scores(Y, oof_a).items()}
    per_a = ({species_list[j]: a for j, a in per_a_idx.items()} if species_list
             else {str(j): a for j, a in per_a_idx.items()})

    result = {
        "readout": "proto",
        "macro_auroc": eval_a["macro_auroc"],        # Strategy A (clip cosine) — primary, always available
        "macro_ap": eval_a["macro_ap"],
        "n_species_evaluated": eval_a["n_evaluated"],
        "n_sites": len(unique_sites),
        "n_windows": int(X.shape[0]),
        "embed_dim": int(X.shape[1]),
        "per_species_auroc": per_a,
        "has_patches": patches is not None,
    }
    if oof_b is not None:
        eval_b = _evaluate(Y, oof_b)
        result["macro_auroc_patch"] = eval_b["macro_auroc"]   # Strategy B (frame max) — the Bird-MAE readout
        result["macro_ap_patch"] = eval_b["macro_ap"]
        result["n_species_evaluated_patch"] = eval_b["n_evaluated"]
    logger.info("proto_loso: clip-cosine AUROC=%.4f%s",
                result["macro_auroc"],
                f", frame-max AUROC={result['macro_auroc_patch']:.4f}" if oof_b is not None else "")
    return result
