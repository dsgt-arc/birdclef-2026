"""Empirical Bayes ridge regression with covariance priors.

Two evaluation tracks (not directly comparable):

**Soundscape track**: train on 739 labeled soundscape windows with LOSO CV.
    Tests: emb+λI, emb+λΣ_emb^{-1}, logit+λI, logit+λΣ_logit^{-1}

**Focal→soundscape transfer track**: train on 265K focal windows, evaluate on
    soundscape per-site. Uses sufficient statistics (X'X, X'Y) accumulated
    per species from partitioned data — never loads all 265K rows at once.

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
from birdclef_106.analysis.eda.covariance_logit import LogitCovariance
from birdclef_106.analysis.eda.covariance_embedding import EmbeddingCovariance
from birdclef_106.analysis.structure.partition import PartitionFeatures
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    list_focal_species,
    load_focal_species,
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


def _ridge_solve_from_stats(XtX, XtY, reg_matrix):
    """Solve ridge from precomputed sufficient statistics."""
    A = XtX + reg_matrix
    return scipy.linalg.solve(A, XtY, assume_a="pos")


def _evaluate(Y_true, Y_pred):
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
    """Ridge regression: two tracks (soundscape LOSO + focal transfer)."""

    def requires(self):
        return {
            "logit_cov": LogitCovariance(**self._params()),
            "emb_cov": EmbeddingCovariance(**self._params()),
            "partition": PartitionFeatures(**self._params()),
        }

    def output(self):
        return [
            self.artifact("ridge_model.npz"),
            self.artifact("ridge_metrics.json"),
            self.artifact("ridge_oof_predictions.npz"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}

        # Load labeled soundscape data (739 rows — small)
        df = load_labeled(self.features_dir)
        sc_emb = embeddings_to_numpy(df).astype(np.float64)
        sc_logits = logits_to_numpy(df).astype(np.float64)
        sc_Y = build_binary_label_matrix(df, species_list).astype(np.float64)
        sc_sites = np.array([extract_site(f) for f in df["filename"].to_list()])
        unique_sites = sorted(set(sc_sites))

        # Load covariances and pre-compute inverses
        logit_cov = np.load(self.input()["logit_cov"].path)["cov"].astype(np.float64)
        emb_cov = np.load(self.input()["emb_cov"].path)["cov"].astype(np.float64)
        logger.info("Ridge: inverting covariance matrices")
        logit_cov_inv = scipy.linalg.inv(logit_cov)
        emb_cov_inv = scipy.linalg.inv(emb_cov)

        results = {}

        # === SOUNDSCAPE TRACK (LOSO CV) ===

        logger.info("Ridge [soundscape]: embeddings + λI")
        results["sc_emb_identity"] = self._cv_ridge(
            sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=None)

        logger.info("Ridge [soundscape]: embeddings + λΣ_emb^{-1}")
        results["sc_emb_empirical_bayes"] = self._cv_ridge(
            sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=emb_cov_inv)

        logger.info("Ridge [soundscape]: logits + λI")
        results["sc_logit_identity"] = self._cv_ridge(
            sc_logits, sc_Y, sc_sites, unique_sites, prior_inv=None)

        logger.info("Ridge [soundscape]: logits + λΣ_logit^{-1}")
        results["sc_logit_empirical_bayes"] = self._cv_ridge(
            sc_logits, sc_Y, sc_sites, unique_sites, prior_inv=logit_cov_inv)

        best_sc_key = max(
            [k for k in results if k.startswith("sc_")],
            key=lambda k: results[k]["best_auroc"])
        logger.info("Ridge [soundscape]: best = %s (AUROC=%.3f)",
                     best_sc_key, results[best_sc_key]["best_auroc"])

        # === FOCAL TRANSFER TRACK (sufficient statistics) ===

        logger.info("Ridge [focal]: accumulating X'X and X'Y from partitioned data")
        available = set(list_focal_species(self.features_dir))
        d_emb = sc_emb.shape[1]  # 1536
        n_sp = len(species_list)

        focal_XtX = np.zeros((d_emb, d_emb), dtype=np.float64)
        focal_XtY = np.zeros((d_emb, n_sp), dtype=np.float64)
        focal_count = 0

        for sp_i, sp in enumerate(species_list):
            if sp not in available:
                continue
            sp_df = load_focal_species(sp, self.features_dir)
            emb = embeddings_to_numpy(sp_df).astype(np.float64)

            focal_XtX += emb.T @ emb
            # Y for this species: column sp_i is all 1s (it's focal data for this species)
            focal_XtY[:, sp_i] += emb.sum(axis=0)
            focal_count += len(sp_df)

            del sp_df, emb
            if (sp_i + 1) % 50 == 0:
                logger.info("  %d/%d species accumulated (%d windows)",
                             sp_i + 1, n_sp, focal_count)

        logger.info("Ridge [focal]: %d total focal windows", focal_count)

        logger.info("Ridge [focal]: embeddings + λI → eval on soundscape")
        results["focal_emb_identity"] = self._focal_ridge_from_stats(
            focal_XtX, focal_XtY, sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=None)

        logger.info("Ridge [focal]: embeddings + λΣ_emb^{-1} → eval on soundscape")
        results["focal_emb_empirical_bayes"] = self._focal_ridge_from_stats(
            focal_XtX, focal_XtY, sc_emb, sc_Y, sc_sites, unique_sites, prior_inv=emb_cov_inv)

        # Best focal config
        focal_keys = [k for k in results if k.startswith("focal_")]
        best_focal_key = max(focal_keys, key=lambda k: results[k]["best_auroc"])
        best_focal = results[best_focal_key]
        logger.info("Ridge [focal]: best = %s (AUROC=%.3f)",
                     best_focal_key, best_focal["best_auroc"])

        focal_prior_inv = emb_cov_inv if "empirical" in best_focal_key else None
        focal_reg = best_focal["best_lambda"] * (
            focal_prior_inv if focal_prior_inv is not None else np.eye(d_emb))
        focal_W = _ridge_solve_from_stats(focal_XtX, focal_XtY, focal_reg)
        focal_oof = sc_emb @ focal_W

        # === SOUNDSCAPE TRACK — save model + OOF ===

        best_sc = results[best_sc_key]
        if "logit" in best_sc_key:
            sc_X_all = sc_logits
            sc_prior_inv = logit_cov_inv if "empirical" in best_sc_key else None
        else:
            sc_X_all = sc_emb
            sc_prior_inv = emb_cov_inv if "empirical" in best_sc_key else None

        d_sc = sc_X_all.shape[1]
        sc_reg = best_sc["best_lambda"] * (
            sc_prior_inv if sc_prior_inv is not None else np.eye(d_sc))
        sc_W = _ridge_solve(sc_X_all, sc_Y, sc_reg)
        sc_oof = self._loso_oof(sc_X_all, sc_Y, sc_sites, unique_sites,
                                best_sc["best_lambda"], sc_prior_inv)

        # === Save ===
        overall_best = best_focal_key if best_focal["best_auroc"] >= best_sc["best_auroc"] else best_sc_key
        primary_W = focal_W if overall_best.startswith("focal") else sc_W
        primary_oof = focal_oof if overall_best.startswith("focal") else sc_oof

        np.savez_compressed(
            self.output()[0].path,
            W=primary_W.astype(np.float32),
            best_lambda=results[overall_best]["best_lambda"],
            feature_space="embeddings",
            config=overall_best,
            species_list=species_list,
            sc_W=sc_W.astype(np.float32),
            sc_config=best_sc_key,
            sc_lambda=best_sc["best_lambda"],
            focal_W=focal_W.astype(np.float32),
            focal_config=best_focal_key,
            focal_lambda=best_focal["best_lambda"],
        )

        np.savez_compressed(
            self.output()[2].path,
            oof_predictions=primary_oof.astype(np.float32),
            config=overall_best,
            species_list=species_list,
            sc_oof=sc_oof.astype(np.float32),
            sc_config=best_sc_key,
            focal_oof=focal_oof.astype(np.float32),
            focal_config=best_focal_key,
        )

        results["_best_soundscape"] = best_sc_key
        results["_best_focal"] = best_focal_key
        results["_primary_model"] = overall_best

        with open(self.output()[1].path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(
            "Ridge: primary=%s (%.3f), sc=%s (%.3f), focal=%s (%.3f)",
            overall_best, results[overall_best]["best_auroc"],
            best_sc_key, best_sc["best_auroc"],
            best_focal_key, best_focal["best_auroc"],
        )

    def _loso_oof(self, X, Y, sites, unique_sites, lam, prior_inv):
        d = X.shape[1]
        reg = lam * (prior_inv if prior_inv is not None else np.eye(d))
        oof = np.zeros_like(Y)
        for site in unique_sites:
            test_mask = sites == site
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            W = _ridge_solve(X[train_mask], Y[train_mask], reg)
            oof[test_mask] = X[test_mask] @ W
        return oof

    def _cv_ridge(self, X, Y, sites, unique_sites, prior_inv=None):
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
            "track": "soundscape",
        }

    def _focal_ridge_from_stats(
        self, XtX, XtY, sc_X, sc_Y, sc_sites, unique_sites, prior_inv=None
    ):
        """Train on focal (from sufficient stats), evaluate per soundscape site."""
        d = XtX.shape[0]
        best_auroc = -1
        best_lambda = None
        all_results = []

        for lam in LAMBDA_GRID:
            reg = lam * (prior_inv if prior_inv is not None else np.eye(d))
            W = _ridge_solve_from_stats(XtX, XtY, reg)
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
            "track": "focal",
        }
