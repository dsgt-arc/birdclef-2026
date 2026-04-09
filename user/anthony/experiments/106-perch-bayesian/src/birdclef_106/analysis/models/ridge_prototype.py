"""Ridge regression on prototype similarity features.

The practical MVP for the star schema prototype framework: use NMF/k-means
prototype similarities as features for multi-label ridge regression. Tests
whether the prototype structure adds value over raw embeddings/logits.

Pipeline:
    1. Load multi-prototype and site-state prototype matrices.
    2. For each labeled soundscape window, compute:
       - Species prototype similarities (R_species,) — cosine to each prototype
       - Site-state prototype similarities (R_site,) — cosine to each site prototype
       - Concatenate → feature vector
    3. Multi-label ridge regression (LOSO CV) on prototype features.
    4. Compare against ridge on raw embeddings (0.749 AUROC baseline).

If prototype features beat raw features, the prototype structure is valuable
and fancier approaches (Hopfield, tensor decomposition) are worth pursuing.

Outputs:
    prototype_ridge_metrics.json — AUROC comparison, per-species breakdown
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.prototypes_species import MultiPrototypeProfiles
from birdclef_106.analysis.structure.prototypes_site import SitePrototypes
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    extract_site,
    load_labeled,
    logits_to_numpy,
    embeddings_to_numpy,
)

logger = logging.getLogger(__name__)


class PrototypeRidge(AnalysisTask):
    """Ridge regression on prototype similarity features (LOSO CV)."""

    def requires(self):
        return [
            MultiPrototypeProfiles(**self._params()),
            SitePrototypes(**self._params()),
        ]

    def output(self):
        return self.artifact("prototype_ridge_metrics.json")

    def run(self):
        self.ensure_parent(self.output())

        species_list = build_species_list(self.artifacts_dir)
        n_species = len(species_list)

        # Load prototypes
        multi_data = np.load(Path(self.artifacts_dir) / "multi_prototypes.npz")
        species_protos = multi_data["prototypes_normed"]  # (R_sp, P)
        proto_species_idx = multi_data["proto_species_idx"]  # (R_sp,)
        global_focal_mean = multi_data["global_focal_mean"]

        site_data = np.load(Path(self.artifacts_dir) / "site_prototypes.npz")
        site_protos = site_data["prototypes_normed"]  # (R_site, P)
        proto_sites = site_data["proto_site"]  # (R_site,)
        global_sc_mean = site_data["global_soundscape_mean"]

        R_sp = species_protos.shape[0]
        R_site = site_protos.shape[0]

        # Load labeled soundscapes
        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_emb = embeddings_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)
        filenames = sc_df["filename"].to_list()
        sites = [extract_site(str(fn)) for fn in filenames]
        n_samples = len(sc_df)

        logger.info("PrototypeRidge: %d samples, %d species protos, %d site protos",
                     n_samples, R_sp, R_site)

        # =====================================================================
        # Build feature matrices
        # =====================================================================

        # Species prototype similarities: cosine(window, each species prototype)
        sc_centered_focal = sc_logits - global_focal_mean[None, :]
        sc_focal_normed = normalize(sc_centered_focal, norm="l2")
        feat_species = sc_focal_normed @ species_protos.T  # (N, R_sp)

        # Site prototype similarities: cosine(window, each site prototype)
        sc_centered_site = sc_logits - global_sc_mean[None, :]
        sc_site_normed = normalize(sc_centered_site, norm="l2")
        feat_site = sc_site_normed @ site_protos.T  # (N, R_site)

        # Raw embeddings (baseline comparison)
        feat_emb = normalize(sc_emb, norm="l2")

        # Feature sets to compare
        feature_sets = {
            "proto_species": feat_species,
            "proto_site": feat_site,
            "proto_combined": np.hstack([feat_species, feat_site]),
            "proto_plus_emb": np.hstack([feat_species, feat_site, feat_emb]),
            "emb_only": feat_emb,
        }

        # =====================================================================
        # LOSO CV (leave-one-site-out)
        # =====================================================================

        unique_sites = sorted(set(sites))
        site_arr = np.array(sites)

        results = {}
        for feat_name, X in feature_sets.items():
            logger.info("PrototypeRidge: evaluating %s (%d features)", feat_name, X.shape[1])

            # Per-species AUROC via LOSO
            all_preds = np.zeros_like(sc_Y, dtype=np.float64)

            for held_out_site in unique_sites:
                test_mask = site_arr == held_out_site
                train_mask = ~test_mask

                if test_mask.sum() == 0 or train_mask.sum() == 0:
                    continue

                X_train, X_test = X[train_mask], X[test_mask]
                Y_train = sc_Y[train_mask]

                # Per-species ridge (independent binary classifiers)
                for si in range(n_species):
                    y_train = Y_train[:, si]
                    if y_train.sum() == 0 or y_train.sum() == len(y_train):
                        # No positive or all positive — predict base rate
                        all_preds[test_mask, si] = y_train.mean()
                        continue

                    clf = RidgeClassifier(alpha=100.0)
                    clf.fit(X_train, y_train)
                    # decision_function gives continuous scores
                    all_preds[test_mask, si] = clf.decision_function(X_test)

            # Compute per-species metrics
            per_species = {}
            aurocs = []
            aps = []
            for si in range(n_species):
                sp = species_list[si]
                y = sc_Y[:, si]
                if y.sum() < 2:
                    continue
                auroc = float(roc_auc_score(y, all_preds[:, si]))
                ap = float(average_precision_score(y, all_preds[:, si]))
                aurocs.append(auroc)
                aps.append(ap)
                per_species[sp] = {"auroc": round(auroc, 4), "ap": round(ap, 4)}

            macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0
            macro_ap = float(np.mean(aps)) if aps else 0.0

            results[feat_name] = {
                "n_features": int(X.shape[1]),
                "macro_auroc": round(macro_auroc, 4),
                "macro_ap": round(macro_ap, 4),
                "n_evaluated": len(aurocs),
                "per_species": per_species,
            }

            logger.info("  %s: AUROC=%.4f, AP=%.4f (%d species)",
                         feat_name, macro_auroc, macro_ap, len(aurocs))

        # Summary comparison
        logger.info("PrototypeRidge: === Summary ===")
        for name, r in results.items():
            logger.info("  %s: AUROC=%.4f (%d features)", name, r["macro_auroc"], r["n_features"])

        with open(self.output().path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("PrototypeRidge: done")
