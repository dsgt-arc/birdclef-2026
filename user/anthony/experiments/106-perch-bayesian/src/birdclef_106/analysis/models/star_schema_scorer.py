"""Star schema scorer: compose species similarity × site affinity.

The star schema treats Perch logit space (P=14795) as the shared dimension
connecting all entities. Scoring a test window decomposes into:

    score(window, species) = sim(window, species_proto)   # what it sounds like
                           × affinity(species, site)       # what's expected here

Each factor is a dot product through P, composed multiplicatively.
Diel variation is captured implicitly by the K multi-prototypes (NMF finds
dawn/night modes as separate components; max-over-K selects the best match).

Evaluation: LOSO CV on 739 labeled soundscape windows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.prototypes_species import MultiPrototypeProfiles
from birdclef_106.analysis.structure.prototypes_site import SitePrototypes
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class StarSchemaScorer(AnalysisTask):
    """Compose species similarity × site affinity through shared Perch logit space."""

    def requires(self):
        p = self._params()
        return {
            "species_protos": MultiPrototypeProfiles(**p),
            "site_protos": SitePrototypes(**p),
        }

    def output(self):
        return [
            self.artifact("star_schema_metrics.json"),
            self.artifact("star_schema_oof.npz"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        n_species = len(species_list)

        # Load labeled soundscape data
        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_emb = embeddings_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)
        filenames = sc_df["filename"].to_list()
        sc_sites = np.array([extract_site(f) for f in filenames])
        unique_sites = sorted(set(sc_sites))

        # Load species prototypes (centered + normalized)
        mp = np.load(str(Path(self.artifacts_dir) / "multi_prototypes.npz"))
        proto_logits_normed = mp["proto_logits_normed"]  # (S, K, P)
        proto_emb_normed = mp["proto_emb_normed"]        # (S, K, E)
        global_focal_mean = mp["global_focal_mean"]       # (P,)
        global_emb_mean = mp["global_emb_mean"]           # (E,)

        # Load site prototypes + affinity
        sp = np.load(str(Path(self.artifacts_dir) / "site_prototypes.npz"))
        affinity_max = sp["species_site_affinity_max"]    # (S, L)
        sites_arr = list(sp["sites"])
        site_to_idx = {s: i for i, s in enumerate(sites_arr)}

        # === Score components ===

        # 1. Species similarity: max cosine over K prototypes
        sc_ln = normalize(sc_logits - global_focal_mean[None, :], norm="l2")
        sc_en = normalize(sc_emb - global_emb_mean[None, :], norm="l2")

        sim_logit = np.einsum("wp,skp->wsk", sc_ln, proto_logits_normed).max(axis=2)
        sim_emb = np.einsum("we,ske->wsk", sc_en, proto_emb_normed).max(axis=2)

        # 2. Site affinity prior: affinity_max[species, site_of_window]
        site_prior = np.zeros((len(sc_df), n_species), dtype=np.float64)
        for i, site in enumerate(sc_sites):
            si = site_to_idx.get(site)
            if si is not None:
                site_prior[i] = affinity_max[:, si]

        # Normalize site prior to [0, 1] per species
        sp_min = site_prior.min(axis=0, keepdims=True)
        sp_max = site_prior.max(axis=0, keepdims=True)
        sp_range = np.maximum(sp_max - sp_min, 1e-8)
        site_prior_norm = (site_prior - sp_min) / sp_range

        # === Compose scores ===
        results = {}

        configs = {
            # Baselines: similarity only
            "logit_sim": sim_logit,
            "emb_sim": sim_emb,
            # Star schema: sim × site (multiplicative blend)
            "logit_site": sim_logit * (0.5 + 0.5 * site_prior_norm),
            "emb_site": sim_emb * (0.5 + 0.5 * site_prior_norm),
            # Additive variant: sim + α*site
            "emb_add_site_01": sim_emb + 0.1 * site_prior_norm,
            "emb_add_site_03": sim_emb + 0.3 * site_prior_norm,
            "emb_add_site_05": sim_emb + 0.5 * site_prior_norm,
        }

        best_auroc = -1
        best_config = None

        for name, scores in configs.items():
            metrics = self._evaluate_loso(scores, sc_Y, sc_sites, unique_sites)
            results[name] = metrics
            logger.info("StarSchema [%s]: AUROC=%.4f, AP=%.4f",
                        name, metrics["macro_auroc"], metrics["macro_ap"])
            if metrics["macro_auroc"] > best_auroc:
                best_auroc = metrics["macro_auroc"]
                best_config = name

        results["_best"] = best_config
        results["_best_auroc"] = best_auroc
        logger.info("StarSchema: best = %s (AUROC=%.4f)", best_config, best_auroc)

        with open(self.output()[0].path, "w") as f:
            json.dump(results, f, indent=2)

        # Save OOF predictions for best config
        np.savez_compressed(
            self.output()[1].path,
            oof_predictions=configs[best_config].astype(np.float32),
            config=best_config,
            species_list=species_list,
            sim_logit=sim_logit.astype(np.float32),
            sim_emb=sim_emb.astype(np.float32),
            site_prior=site_prior_norm.astype(np.float32),
        )

    def _evaluate_loso(self, scores, Y, sites, unique_sites):
        """Evaluate per-site AUROC, then average across sites."""
        fold_aurocs, fold_aps = [], []
        for site in unique_sites:
            mask = sites == site
            if mask.sum() == 0:
                continue
            Y_fold = Y[mask]
            S_fold = scores[mask]
            aurocs, aps = [], []
            for j in range(Y.shape[1]):
                y = Y_fold[:, j]
                if y.sum() < 2 or y.sum() >= len(y) - 1:
                    continue
                aurocs.append(roc_auc_score(y, S_fold[:, j]))
                aps.append(average_precision_score(y, S_fold[:, j]))
            if aurocs:
                fold_aurocs.append(float(np.mean(aurocs)))
                fold_aps.append(float(np.mean(aps)))

        return {
            "macro_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else 0.0,
            "macro_ap": float(np.mean(fold_aps)) if fold_aps else 0.0,
            "n_folds": len(fold_aurocs),
        }
