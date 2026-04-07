"""Empirical Bayes ridge regression with covariance priors.

Standard ridge uses λI as regularization, treating all feature directions equally.
Empirical Bayes ridge uses λΣ^{-1} where Σ is the covariance estimated from
127K unlabeled soundscape windows. This gives better regularization: directions
where features vary a lot (high variance in Σ) get less penalized, directions with
low variance get shrunk hard.

Compares 4 configurations:
    1. Embeddings + λI (standard ridge, 1536-d)
    2. Embeddings + λΣ_emb^{-1} (empirical Bayes, 1536-d)
    3. Logits + λI (standard ridge, 14795-d — high p/n)
    4. Logits + λΣ_logit^{-1} (empirical Bayes, 14795-d)

Also trains on focal data (265K windows) as an alternative to soundscape-only,
with soundscape LOSO as the evaluation.

References:
    Tang et al. 2020 — Deep Transfer Learning with Ridge Regression
    Munch et al. 2022 — Semi-supervised empirical Bayes factor regression
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.linalg
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.logit_covariance import LogitCovariance
from birdclef_106.analysis.embedding_covariance import EmbeddingCovariance
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    load_focal_shards,
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
    """Ridge regression comparing feature spaces, priors, and training data."""

    def requires(self):
        return {
            "logit_cov": LogitCovariance(**self._params()),
            "emb_cov": EmbeddingCovariance(**self._params()),
        }

    def output(self):
        return [
            self.artifact("ridge_model.npz"),
            self.artifact("ridge_metrics.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)

        # Load labeled soundscape data
        df = load_labeled(self.features_dir)
        sc_emb = embeddings_to_numpy(df).astype(np.float64)   # (739, 1536)
        sc_logits = logits_to_numpy(df).astype(np.float64)     # (739, 14795)
        sc_Y = build_binary_label_matrix(df, species_list).astype(np.float64)
        sc_sites = np.array([extract_site(f) for f in df["filename"].to_list()])
        unique_sites = sorted(set(sc_sites))

        # Load covariances
        logit_cov = np.load(self.input()["logit_cov"].path)["cov"].astype(np.float64)
        emb_cov = np.load(self.input()["emb_cov"].path)["cov"].astype(np.float64)

        # Pre-compute inverses once (expensive for logit cov)
        logger.info("Ridge: inverting covariance matrices")
        logit_cov_inv = scipy.linalg.inv(logit_cov)
        emb_cov_inv = scipy.linalg.inv(emb_cov)

        results = {}

        # --- Soundscape-only training, 4 configurations ---

        logger.info("Ridge: embeddings + λI")
        results["emb_identity"] = self._cv_ridge(
            sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=None)

        logger.info("Ridge: embeddings + λΣ_emb^{-1}")
        results["emb_empirical_bayes"] = self._cv_ridge(
            sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=emb_cov_inv)

        logger.info("Ridge: logits + λI")
        results["logit_identity"] = self._cv_ridge(
            sc_logits, sc_Y, sc_sites, unique_sites, prior_inv=None)

        logger.info("Ridge: logits + λΣ_logit^{-1}")
        results["logit_empirical_bayes"] = self._cv_ridge(
            sc_logits, sc_Y, sc_sites, unique_sites, prior_inv=logit_cov_inv)

        # --- Focal-trained, evaluated on soundscape ---

        logger.info("Ridge: loading focal data for focal-trained models")
        focal_emb_parts, focal_Y_parts = [], []
        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            focal_emb_parts.append(embeddings_to_numpy(shard).astype(np.float64))
            focal_Y_parts.append(
                build_binary_label_matrix(shard, species_list).astype(np.float64))
            if i % 4 == 0:
                logger.info("  focal shard %d loaded", i)

        focal_emb = np.concatenate(focal_emb_parts)
        focal_Y = np.concatenate(focal_Y_parts)
        del focal_emb_parts, focal_Y_parts

        logger.info("Ridge: focal-trained embeddings + λI → eval on soundscape")
        results["focal_emb_identity"] = self._focal_ridge(
            focal_emb, focal_Y, sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=None)

        logger.info("Ridge: focal-trained embeddings + λΣ_emb^{-1} → eval on soundscape")
        results["focal_emb_empirical_bayes"] = self._focal_ridge(
            focal_emb, focal_Y, sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=emb_cov_inv)

        del focal_emb, focal_Y

        # --- Pick best model and save ---

        best_key = max(results, key=lambda k: results[k].get("best_auroc", 0))
        best = results[best_key]
        logger.info("Ridge: best config = %s (AUROC=%.3f)", best_key, best["best_auroc"])

        # Retrain best on all soundscape data
        if "logit" in best_key:
            X_all = sc_logits
            prior_inv = logit_cov_inv if "empirical" in best_key else None
        else:
            X_all = sc_emb
            prior_inv = emb_cov_inv if "empirical" in best_key else None

        d = X_all.shape[1]
        reg = best["best_lambda"] * (prior_inv if prior_inv is not None else np.eye(d))
        W = _ridge_solve(X_all, sc_Y, reg)

        np.savez_compressed(
            self.output()[0].path,
            W=W.astype(np.float32),
            best_lambda=best["best_lambda"],
            feature_space="logits" if "logit" in best_key else "embeddings",
            config=best_key,
            species_list=species_list,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Ridge: saved model (%s) and metrics", best_key)

    def _cv_ridge(self, X, Y, sites, unique_sites, prior_inv=None):
        """LOSO cross-validation on soundscape data."""
        d = X.shape[1]
        best_auroc = -1
        best_lambda = None
        all_results = []

        for lam in LAMBDA_GRID:
            reg = lam * (prior_inv if prior_inv is not None else np.eye(d))
            fold_aurocs, fold_aps = [], []

            for site in unique_sites:
                test_mask = sites == site
                train_mask = ~test_mask
                if test_mask.sum() == 0 or train_mask.sum() == 0:
                    continue
                W = _ridge_solve(X[train_mask], Y[train_mask], reg)
                metrics = _evaluate(Y[test_mask], X[test_mask] @ W)
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
            "best_lambda": best_lambda, "best_auroc": best_auroc,
            "best_ap": next((r["ap"] for r in all_results if r["lambda"] == best_lambda), 0.0),
            "lambda_sweep": all_results, "n_features": d,
            "train_data": "soundscape",
        }

    def _focal_ridge(self, focal_X, focal_Y, sc_X, sc_Y, sc_sites, unique_sites, prior_inv=None):
        """Train on focal, evaluate on soundscape LOSO (held-out site = test)."""
        d = focal_X.shape[1]
        best_auroc = -1
        best_lambda = None
        all_results = []

        for lam in LAMBDA_GRID:
            reg = lam * (prior_inv if prior_inv is not None else np.eye(d))
            # Train on ALL focal data (no focal CV — soundscape sites are the folds)
            W = _ridge_solve(focal_X, focal_Y, reg)
            # Evaluate on each soundscape site
            fold_aurocs, fold_aps = [], []
            for site in unique_sites:
                test_mask = sc_sites == site
                if test_mask.sum() == 0:
                    continue
                metrics = _evaluate(sc_Y[test_mask], sc_X[test_mask] @ W)
                if metrics["n_evaluated"] > 0:
                    fold_aurocs.append(metrics["macro_auroc"])
                    fold_aps.append(metrics["macro_ap"])

            mean_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
            mean_ap = float(np.mean(fold_aps)) if fold_aps else 0.0
            all_results.append({"lambda": lam, "auroc": mean_auroc, "ap": mean_ap})
            if mean_auroc > best_auroc:
                best_auroc = mean_auroc
                best_lambda = lam
            logger.info("  λ=%.4f: AUROC=%.3f, AP=%.3f (focal→soundscape)", lam, mean_auroc, mean_ap)

        return {
            "best_lambda": best_lambda, "best_auroc": best_auroc,
            "best_ap": next((r["ap"] for r in all_results if r["lambda"] == best_lambda), 0.0),
            "lambda_sweep": all_results, "n_features": d,
            "train_data": "focal",
        }
