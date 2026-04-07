"""Sparse logistic regression: per-species L1-regularized on Perch logits.

For each BirdCLEF target species, fits an L1-regularized logistic regression
on the 14,795 Perch logits. The L1 penalty produces sparse weights, revealing
which specific Perch species logits predict each BirdCLEF target.

Interpretable: "BirdCLEF species X is predicted by Perch logits for species
A (positive) and B (negative)." This can reveal acoustic similarity (same
species under different taxonomy), co-occurrence, or confusion patterns.

Complementary to ridge regression — ridge gives the best prediction,
sparse logistic gives the most interpretable model.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.logit_covariance import LogitCovariance
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    extract_site,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


class SparseLogistic(AnalysisTask):
    """Per-species L1 logistic regression on Perch logits with LOSO CV."""

    def requires(self):
        return []  # no dependencies — operates on raw labeled data

    def output(self):
        return [
            self.artifact("sparse_logistic_model.npz"),
            self.artifact("sparse_logistic_metrics.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        df = load_labeled(self.features_dir)
        X = logits_to_numpy(df).astype(np.float64)  # (739, 14795)
        Y = build_binary_label_matrix(df, species_list)  # (739, 234)

        sites = np.array([extract_site(f) for f in df["filename"].to_list()])
        unique_sites = sorted(set(sites))

        per_species = {}
        all_coefs = {}

        for si, sp in enumerate(species_list):
            y = Y[:, si]
            n_pos = int(y.sum())
            if n_pos < 5 or n_pos == len(y) - 1:
                per_species[sp] = {"auroc": None, "ap": None, "n_pos": n_pos, "note": "too few positives"}
                continue

            # LOSO CV to select C
            best_c, best_auroc = None, -1
            for C in C_GRID:
                fold_aurocs = []
                for site in unique_sites:
                    test_mask = sites == site
                    train_mask = ~test_mask
                    if y[train_mask].sum() < 1 or y[test_mask].sum() < 1:
                        continue

                    clf = LogisticRegression(
                        penalty="l1", C=C, solver="liblinear", max_iter=1000, random_state=42,
                    )
                    clf.fit(X[train_mask], y[train_mask])
                    y_pred = clf.predict_proba(X[test_mask])[:, 1]
                    try:
                        fold_aurocs.append(roc_auc_score(y[test_mask], y_pred))
                    except ValueError:
                        continue

                mean_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
                if mean_auroc > best_auroc:
                    best_auroc = mean_auroc
                    best_c = C

            if best_c is None:
                per_species[sp] = {"auroc": None, "ap": None, "n_pos": n_pos, "note": "no valid CV folds"}
                continue

            # Refit on all data with best C
            clf = LogisticRegression(
                penalty="l1", C=best_c, solver="liblinear", max_iter=1000, random_state=42,
            )
            clf.fit(X, y)
            y_pred_all = clf.predict_proba(X)[:, 1]

            coef = clf.coef_.ravel()
            n_nonzero = int((coef != 0).sum())
            top_pos = np.argsort(-coef)[:10]
            top_neg = np.argsort(coef)[:10]

            per_species[sp] = {
                "auroc": best_auroc,
                "ap": float(average_precision_score(y, y_pred_all)),
                "n_pos": n_pos,
                "best_C": best_c,
                "n_nonzero_coefs": n_nonzero,
                "top_positive_logits": [
                    {"logit_index": int(j), "coef": float(coef[j])} for j in top_pos if coef[j] > 0
                ],
                "top_negative_logits": [
                    {"logit_index": int(j), "coef": float(coef[j])} for j in top_neg if coef[j] < 0
                ],
            }
            all_coefs[sp] = coef

            if (si + 1) % 20 == 0:
                logger.info("SparseLogistic: %d/%d species done", si + 1, len(species_list))

        aurocs = [v["auroc"] for v in per_species.values() if v["auroc"] is not None]
        aps = [v["ap"] for v in per_species.values() if v["ap"] is not None]

        summary = {
            "per_species": per_species,
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "macro_ap": float(np.mean(aps)) if aps else None,
            "n_evaluated": len(aurocs),
        }

        # Save coefficients as sparse-ish npz
        coef_matrix = np.zeros((len(species_list), X.shape[1]), dtype=np.float32)
        for si, sp in enumerate(species_list):
            if sp in all_coefs:
                coef_matrix[si] = all_coefs[sp]

        np.savez_compressed(
            self.output()[0].path,
            coef_matrix=coef_matrix,
            species_list=species_list,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "SparseLogistic: macro AUROC=%.3f, macro AP=%.3f (%d species)",
            summary["macro_auroc"] or 0, summary["macro_ap"] or 0, summary["n_evaluated"],
        )
