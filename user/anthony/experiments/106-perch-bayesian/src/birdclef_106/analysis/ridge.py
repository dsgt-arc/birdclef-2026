"""Empirical Bayes ridge regression with logit covariance prior.

Standard ridge uses λI as regularization, treating all feature directions equally.
Empirical Bayes ridge uses λΣ^{-1} where Σ is the logit covariance estimated from
127K unlabeled soundscape windows. This gives better regularization: directions
where logits vary a lot (high variance in Σ) get less penalized, directions with
low variance get shrunk hard.

The semi-supervised aspect: unlabeled data informs the prior via covariance,
labeled data (739 windows) provides the likelihood.

Implements both feature spaces:
    - Option A: ridge on embeddings (1536-d) with standard λI prior
    - Option B: ridge on logits (14795-d) with empirical Bayes λΣ^{-1} prior

References:
    Tang et al. 2020 — Deep Transfer Learning with Ridge Regression
    Munch et al. 2022 — Semi-supervised empirical Bayes factor regression
"""

from __future__ import annotations

import json
import logging

import numpy as np
import scipy.linalg
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.logit_covariance import LogitCovariance
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]


def _ridge_solve(X, Y, reg_matrix):
    """Solve ridge: W = (X'X + reg_matrix)^{-1} X'Y via scipy."""
    A = X.T @ X + reg_matrix
    B = X.T @ Y
    return scipy.linalg.solve(A, B, assume_a="pos")


def _evaluate(Y_true, Y_pred):
    """Per-species AUROC and AP, skipping degenerate cases."""
    n_species = Y_true.shape[1]
    aurocs, aps = [], []
    for j in range(n_species):
        y = Y_true[:, j]
        if y.sum() < 2 or y.sum() >= len(y) - 1:
            continue
        aurocs.append(roc_auc_score(y, Y_pred[:, j]))
        aps.append(average_precision_score(y, Y_pred[:, j]))
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "macro_ap": float(np.mean(aps)) if aps else 0.0,
        "n_evaluated": len(aurocs),
    }


class EmpiricalBayesRidge(AnalysisTask):
    """Ridge regression with LOSO CV, comparing embedding vs logit feature spaces."""

    def requires(self):
        return LogitCovariance(**self._params())

    def output(self):
        return [
            self.artifact("ridge_model.npz"),
            self.artifact("ridge_metrics.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        df = load_labeled(self.features_dir)
        emb = embeddings_to_numpy(df)  # (739, 1536)
        logits = logits_to_numpy(df)   # (739, 14795)
        Y = build_binary_label_matrix(df, species_list).astype(np.float64)

        sites = np.array([extract_site(f) for f in df["filename"].to_list()])
        unique_sites = sorted(set(sites))

        # Load logit covariance for empirical Bayes prior
        cov_data = np.load(self.input().path)
        logit_cov = cov_data["cov"].astype(np.float64)

        results = {}

        # Option A: Ridge on embeddings with standard λI
        logger.info("Ridge: evaluating embeddings (1536-d) with λI prior")
        best_emb = self._cv_ridge(emb.astype(np.float64), Y, sites, unique_sites, prior="identity")
        results["embeddings_identity"] = best_emb

        # Option B: Ridge on logits with empirical Bayes λΣ^{-1}
        logger.info("Ridge: evaluating logits (14795-d) with λΣ^{-1} prior")
        best_logit = self._cv_ridge(
            logits.astype(np.float64), Y, sites, unique_sites,
            prior="covariance", cov=logit_cov,
        )
        results["logits_empirical_bayes"] = best_logit

        # Pick best overall model, retrain on all data
        if (best_emb["best_auroc"] or 0) >= (best_logit["best_auroc"] or 0):
            best_space = "embeddings"
            X_all = emb.astype(np.float64)
            reg = best_emb["best_lambda"] * np.eye(X_all.shape[1])
        else:
            best_space = "logits"
            X_all = logits.astype(np.float64)
            cov_inv = scipy.linalg.inv(logit_cov)
            reg = best_logit["best_lambda"] * cov_inv

        W = _ridge_solve(X_all, Y, reg)

        np.savez_compressed(
            self.output()[0].path,
            W=W.astype(np.float32),
            best_lambda=results[f"{best_space}_{'identity' if best_space == 'embeddings' else 'empirical_bayes'}"]["best_lambda"],
            feature_space=best_space,
            species_list=species_list,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(
            "Ridge: best=%s, AUROC=%.3f, AP=%.3f, λ=%.4f",
            best_space,
            results[f"{best_space}_{'identity' if best_space == 'embeddings' else 'empirical_bayes'}"]["best_auroc"],
            results[f"{best_space}_{'identity' if best_space == 'embeddings' else 'empirical_bayes'}"]["best_ap"],
            results[f"{best_space}_{'identity' if best_space == 'embeddings' else 'empirical_bayes'}"]["best_lambda"],
        )

    def _cv_ridge(self, X, Y, sites, unique_sites, prior="identity", cov=None):
        """LOSO cross-validation over lambda grid."""
        d = X.shape[1]

        if prior == "covariance" and cov is not None:
            cov_inv = scipy.linalg.inv(cov)
        else:
            cov_inv = None

        best_auroc = -1
        best_lambda = None
        all_results = []

        for lam in LAMBDA_GRID:
            if cov_inv is not None:
                reg = lam * cov_inv
            else:
                reg = lam * np.eye(d)

            fold_aurocs = []
            fold_aps = []

            for site in unique_sites:
                test_mask = sites == site
                train_mask = ~test_mask
                if test_mask.sum() == 0 or train_mask.sum() == 0:
                    continue

                W = _ridge_solve(X[train_mask], Y[train_mask], reg)
                Y_pred = X[test_mask] @ W

                metrics = _evaluate(Y[test_mask], Y_pred)
                if metrics["n_evaluated"] > 0:
                    fold_aurocs.append(metrics["macro_auroc"])
                    fold_aps.append(metrics["macro_ap"])

            mean_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
            mean_ap = float(np.mean(fold_aps)) if fold_aps else 0.0
            all_results.append({"lambda": lam, "auroc": mean_auroc, "ap": mean_ap})

            if mean_auroc > best_auroc:
                best_auroc = mean_auroc
                best_lambda = lam

            logger.info("  λ=%.4f: AUROC=%.3f, AP=%.3f", lam, mean_auroc, mean_ap)

        return {
            "best_lambda": best_lambda,
            "best_auroc": best_auroc,
            "best_ap": next((r["ap"] for r in all_results if r["lambda"] == best_lambda), 0.0),
            "lambda_sweep": all_results,
            "prior": prior,
            "n_features": d,
        }
