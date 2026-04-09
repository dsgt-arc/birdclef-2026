"""Star schema scorer: compose species similarity × site affinity.

The star schema treats Perch logit space (P=14795) as the shared dimension
connecting all entities. Scoring a test window decomposes into:

    score(window, species) = sim(window, species_proto)   # what it sounds like
                           × affinity(species, site)       # what's expected here

All dot products are weighted by domain transfer precision:

    transfer_weight_d = reliability_d × specificity_d × decorrelation_d

- Reliability: (1 - ks_stat_d) — downweight dimensions that shift focal→soundscape
- Specificity: 1/sqrt(n_species_activating_d) — upweight rare discriminative dims
- Decorrelation: Mahalanobis whitening via covariance eigenvectors

Diel variation is captured implicitly by K multi-prototypes (NMF finds dawn/night
modes as separate components; max-over-K selects the best match).

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
from birdclef_106.analysis.eda.covariance_logit import LogitCovariance
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


def _build_transfer_weights(artifacts_dir: str) -> np.ndarray:
    """Build per-dimension transfer weights from cached artifacts.

    Returns (14795,) weight vector. Higher = more trustworthy for cross-domain
    composition through logit space.
    """
    artifacts = Path(artifacts_dir)

    # 1. Reliability: (1 - ks_stat) — stable dimensions get high weight
    ds = np.load(str(artifacts / "distribution_shift.npz"))
    ks_stats = ds["ks_stats"].astype(np.float64)
    reliability = np.maximum(1.0 - ks_stats, 0.0)

    # 2. Specificity: 1/sqrt(n_species_activating) — rare dims more discriminative
    fp = np.load(str(artifacts / "focal_profiles_raw.npz"))
    act_freq = fp["activation_freq"]  # (234, 14795)
    n_activating = (act_freq > 0.1).sum(axis=0).astype(np.float64)
    specificity = 1.0 / np.sqrt(np.maximum(n_activating, 1.0))
    # Dims activated by 0 species get zero weight (no signal)
    specificity[n_activating == 0] = 0.0

    # 3. Decorrelation: project through top-k eigenvectors, scale by 1/sqrt(eigenvalue)
    #    This whitens the space so correlated logit clusters contribute as ~1 dimension.
    #    Applied as a diagonal approximation: weight_d = sum_k(V_dk^2 / lambda_k)
    cd = np.load(str(artifacts / "logit_covariance_derived.npz"))
    eigenvectors = cd["eigenvectors"].astype(np.float64)  # (14795, 500)
    eigenvalues = cd["eigenvalues"].astype(np.float64)     # (500,)
    inv_sqrt_eig = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-4))
    # Diagonal of V @ diag(1/sqrt(λ)) @ V.T — per-dimension decorrelation factor
    decorrelation = (eigenvectors ** 2 @ inv_sqrt_eig)

    # Combine multiplicatively
    weights = reliability * specificity * decorrelation

    # Normalize to unit L1 for interpretable scaling
    weights = weights / np.maximum(weights.sum(), 1e-8)

    logger.info(
        "Transfer weights: reliability=[%.3f,%.3f], specificity=[%.3f,%.3f], "
        "decorrelation=[%.3f,%.3f], combined nonzero=%d/%d",
        reliability.min(), reliability.max(),
        specificity.min(), specificity.max(),
        decorrelation.min(), decorrelation.max(),
        (weights > 1e-10).sum(), len(weights),
    )

    return weights


class StarSchemaScorer(AnalysisTask):
    """Compose species similarity × site affinity with transfer-weighted logit space."""

    def requires(self):
        p = self._params()
        return {
            "species_protos": MultiPrototypeProfiles(**p),
            "site_protos": SitePrototypes(**p),
            "logit_cov": LogitCovariance(**p),
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

        # Load species prototypes
        mp = np.load(str(Path(self.artifacts_dir) / "multi_prototypes.npz"))
        proto_logits = mp["proto_logits"]              # (S, K, P) raw centered
        proto_emb_normed = mp["proto_emb_normed"]      # (S, K, E) centered+normed
        global_focal_mean = mp["global_focal_mean"]     # (P,)
        global_emb_mean = mp["global_emb_mean"]         # (E,)
        K_species = int(mp["n_prototypes"])

        # Load site prototypes
        sp = np.load(str(Path(self.artifacts_dir) / "site_prototypes.npz"))
        site_proto_logits = sp["proto_logits"]          # (L, K_site, P) raw centered
        sites_arr = list(sp["sites"])
        site_to_idx = {s: i for i, s in enumerate(sites_arr)}
        global_sc_mean = sp["global_soundscape_mean"]

        # Build transfer weights
        transfer_w = _build_transfer_weights(self.artifacts_dir)  # (P,)
        sqrt_w = np.sqrt(transfer_w)  # for weighted cosine: w-dot(a,b) = dot(sqrt(w)*a, sqrt(w)*b)

        # === Weighted logit-space scoring ===

        # Center test windows and apply transfer weights
        sc_logits_c = sc_logits - global_focal_mean[None, :]
        sc_logits_w = sc_logits_c * sqrt_w[None, :]
        sc_logits_wn = normalize(sc_logits_w, norm="l2")

        # Weight + normalize species prototypes
        proto_logits_c = proto_logits - global_focal_mean[None, None, :]
        proto_logits_w = proto_logits_c * sqrt_w[None, None, :]
        proto_logits_wn = proto_logits_w / np.maximum(
            np.linalg.norm(proto_logits_w, axis=2, keepdims=True), 1e-8)

        # Weighted cosine similarity: max over K prototypes
        sim_logit_weighted = np.einsum("wp,skp->wsk", sc_logits_wn, proto_logits_wn).max(axis=2)

        # Unweighted logit similarity (baseline)
        sc_ln = normalize(sc_logits_c, norm="l2")
        proto_ln = mp["proto_logits_normed"]
        sim_logit_raw = np.einsum("wp,skp->wsk", sc_ln, proto_ln).max(axis=2)

        # Embedding similarity (no transfer weighting — embeddings are 1536-d, no per-dim shift data)
        sc_en = normalize(sc_emb - global_emb_mean[None, :], norm="l2")
        sim_emb = np.einsum("we,ske->wsk", sc_en, proto_emb_normed).max(axis=2)

        # === Weighted site affinity ===

        # Recompute affinity with transfer weights
        # Species profiles: mean over prototypes, weighted
        sp_profiles_w = (proto_logits_c.mean(axis=1)) * sqrt_w[None, :]
        sp_profiles_wn = normalize(sp_profiles_w, norm="l2")

        # Site prototypes: center by soundscape mean, weight
        site_proto_c = site_proto_logits - global_sc_mean[None, None, :]
        site_proto_w = site_proto_c * sqrt_w[None, None, :]
        site_proto_wn = site_proto_w / np.maximum(
            np.linalg.norm(site_proto_w, axis=2, keepdims=True), 1e-8)

        # Affinity: (S, P) @ (L, K_site, P).T → max over K_site
        n_sites = len(sites_arr)
        K_site = site_proto_wn.shape[1]
        flat_site = site_proto_wn.reshape(-1, site_proto_wn.shape[2])  # (L*K_site, P)
        affinity_flat = sp_profiles_wn @ flat_site.T  # (S, L*K_site)
        affinity_w = affinity_flat.reshape(n_species, n_sites, K_site).max(axis=2)  # (S, L)

        # Also compute unweighted affinity for comparison
        affinity_raw = sp["species_site_affinity_max"]

        # Build per-window site prior (weighted and raw)
        def _site_prior_matrix(affinity):
            prior = np.zeros((len(sc_df), n_species), dtype=np.float64)
            for i, site in enumerate(sc_sites):
                si = site_to_idx.get(site)
                if si is not None:
                    prior[i] = affinity[:, si]
            sp_min = prior.min(axis=0, keepdims=True)
            sp_max = prior.max(axis=0, keepdims=True)
            sp_range = np.maximum(sp_max - sp_min, 1e-8)
            return (prior - sp_min) / sp_range

        site_prior_w = _site_prior_matrix(affinity_w)
        site_prior_raw = _site_prior_matrix(affinity_raw)

        # === Compose scores ===
        results = {}

        configs = {
            # Baselines: similarity only
            "logit_raw": sim_logit_raw,
            "logit_weighted": sim_logit_weighted,
            "emb_sim": sim_emb,
            # Logit weighted + site (weighted affinity)
            "logit_w_site_w": sim_logit_weighted * (0.5 + 0.5 * site_prior_w),
            # Logit weighted + site (raw affinity, for comparison)
            "logit_w_site_raw": sim_logit_weighted * (0.5 + 0.5 * site_prior_raw),
            # Embedding + site (weighted affinity)
            "emb_site_w": sim_emb * (0.5 + 0.5 * site_prior_w),
            # Additive variants
            "logit_w_add_site_03": sim_logit_weighted + 0.3 * site_prior_w,
            "emb_add_site_w_03": sim_emb + 0.3 * site_prior_w,
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

        np.savez_compressed(
            self.output()[1].path,
            oof_predictions=configs[best_config].astype(np.float32),
            config=best_config,
            species_list=species_list,
            sim_logit_raw=sim_logit_raw.astype(np.float32),
            sim_logit_weighted=sim_logit_weighted.astype(np.float32),
            sim_emb=sim_emb.astype(np.float32),
            site_prior_w=site_prior_w.astype(np.float32),
            site_prior_raw=site_prior_raw.astype(np.float32),
            transfer_weights=transfer_w.astype(np.float32),
            affinity_weighted=affinity_w.astype(np.float32),
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
