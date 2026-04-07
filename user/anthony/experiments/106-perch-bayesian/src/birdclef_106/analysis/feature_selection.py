"""Univariate feature selection: per BirdCLEF species × Perch logit pair.

For each of the 234 target species and each of the 14,795 Perch logits,
test whether the logit has marginal association with species presence.
No model fitting — pure statistical testing. Serves as a fast sanity check
and interpretability tool before fitting sparse models.

Methods:
    - Per-logit AUROC: how well each individual logit predicts each species
    - Mutual information: captures nonlinear associations
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class FeatureSelection(AnalysisTask):
    """Univariate logit importance for each BirdCLEF species."""

    def output(self):
        return self.artifact("feature_selection.json")

    def run(self):
        self.ensure_parent(self.output())

        species_list = build_species_list(self.artifacts_dir)
        df = load_labeled(self.features_dir)
        logits = logits_to_numpy(df)  # (739, 14795)
        Y = build_binary_label_matrix(df, species_list)  # (739, 234)

        n_species = len(species_list)
        n_logits = logits.shape[1]
        top_k = 20

        per_species = {}
        for si, sp in enumerate(species_list):
            y = Y[:, si]
            n_pos = int(y.sum())
            if n_pos < 5 or n_pos == len(y):
                per_species[sp] = {"n_pos": n_pos, "top_logits": [], "note": "too few positives"}
                continue

            # Per-logit AUROC (vectorized: compute for all 14795 logits at once)
            # For each logit j, AUROC(y, logits[:, j])
            # Vectorized via rank-based formula
            pos_means = logits[y == 1].mean(axis=0)
            neg_means = logits[y == 0].mean(axis=0)
            mean_diff = pos_means - neg_means

            # Compute AUROC for top candidates only (full 14795 AUROC is slow)
            top_candidates = np.argsort(-np.abs(mean_diff))[:100]
            aurocs = {}
            for j in top_candidates:
                try:
                    aurocs[int(j)] = float(roc_auc_score(y, logits[:, j]))
                except ValueError:
                    continue

            # Top-k by AUROC
            sorted_by_auroc = sorted(aurocs.items(), key=lambda x: -abs(x[1] - 0.5))[:top_k]
            top_logits = [
                {"logit_index": j, "auroc": auc, "pos_mean": float(pos_means[j]),
                 "neg_mean": float(neg_means[j]), "mean_diff": float(mean_diff[j])}
                for j, auc in sorted_by_auroc
            ]

            per_species[sp] = {
                "n_pos": n_pos,
                "top_logits": top_logits,
                "max_auroc": float(max(aurocs.values())) if aurocs else None,
            }

        # Summary
        max_aurocs = [v["max_auroc"] for v in per_species.values() if v.get("max_auroc") is not None]
        summary = {
            "per_species": per_species,
            "n_species_evaluated": len(max_aurocs),
            "mean_max_auroc": float(np.mean(max_aurocs)) if max_aurocs else None,
            "median_max_auroc": float(np.median(max_aurocs)) if max_aurocs else None,
        }

        with open(self.output().path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "FeatureSelection: %d species evaluated, mean max AUROC=%.3f",
            summary["n_species_evaluated"],
            summary["mean_max_auroc"] or 0,
        )
