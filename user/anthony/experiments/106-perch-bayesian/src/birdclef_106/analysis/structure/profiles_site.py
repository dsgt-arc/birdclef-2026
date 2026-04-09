"""Site residual profiles with James-Stein shrinkage and Mahalanobis weighting.

Uses partitioned by_site/ parquets — loads one site at a time instead of
scanning all 16 unlabeled shards (7 GB). Peak memory: ~2.5 GB (one large site).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import linalg

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.partition import PartitionFeatures
from birdclef_106.data_loading import (
    extract_site,
    list_sites,
    load_labeled,
    load_site,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
MAHAL_REG = 1e-4
TOP_K_SUMMARY = 30
MAX_QUANTILE_SAMPLES = 10_000


class SiteResidualProfiles(AnalysisTask):
    """Residual + shrinkage + Mahalanobis site profiles from unlabeled soundscapes."""

    def requires(self):
        return PartitionFeatures(**self._params())

    def output(self):
        return [
            self.artifact("site_residuals.npz"),
            self.artifact("site_residuals_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        perch_labels = self._load_perch_labels()
        sites = list_sites(self.features_dir)
        n_sites = len(sites)

        # === Pass 1: per-site sufficient statistics ===
        logger.info("SiteResiduals: computing stats for %d sites", n_sites)

        logit_dim = None
        site_sum_map: dict[str, np.ndarray] = {}
        site_sum_sq_map: dict[str, np.ndarray] = {}
        site_count_map: dict[str, int] = {}

        global_sum = None
        global_sum_sq = None
        global_count = 0

        for si, site in enumerate(sites):
            df = load_site(site, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)

            if logit_dim is None:
                logit_dim = logits.shape[1]
                global_sum = np.zeros(logit_dim, dtype=np.float64)
                global_sum_sq = np.zeros(logit_dim, dtype=np.float64)

            s = logits.sum(axis=0)
            sq = (logits ** 2).sum(axis=0)
            n = len(df)

            site_sum_map[site] = s
            site_sum_sq_map[site] = sq
            site_count_map[site] = n

            global_sum += s
            global_sum_sq += sq
            global_count += n

            del df, logits
            if (si + 1) % 5 == 0:
                logger.info("  %d/%d sites done", si + 1, n_sites)

        # Also include labeled soundscapes in global stats
        labeled = load_labeled(self.features_dir)
        lab_logits = logits_to_numpy(labeled).astype(np.float64)
        global_sum += lab_logits.sum(axis=0)
        global_sum_sq += (lab_logits ** 2).sum(axis=0)
        global_count += len(labeled)

        # Per-site stats from labeled data
        for fn, row_logits in zip(labeled["filename"].to_list(), lab_logits):
            site = extract_site(fn)
            if site in site_sum_map:
                site_sum_map[site] += row_logits
                site_sum_sq_map[site] += row_logits ** 2
                site_count_map[site] += 1

        del labeled, lab_logits

        logger.info("SiteResiduals: %d sites, %d total windows", n_sites, global_count)

        # === Compute means, variances, residuals ===
        global_mean = global_sum / global_count
        global_var = global_sum_sq / global_count - global_mean ** 2

        site_means = np.zeros((n_sites, logit_dim), dtype=np.float64)
        site_vars = np.zeros((n_sites, logit_dim), dtype=np.float64)
        site_counts = np.array([site_count_map[s] for s in sites])

        for si, site in enumerate(sites):
            n = site_count_map[site]
            site_means[si] = site_sum_map[site] / n
            site_vars[si] = site_sum_sq_map[site] / n - site_means[si] ** 2

        residuals = site_means - global_mean[None, :]

        # === James-Stein shrinkage ===
        logger.info("SiteResiduals: computing James-Stein shrinkage")

        shrinkage_alpha = np.zeros(n_sites, dtype=np.float64)
        shrunk_residuals = np.zeros_like(residuals)

        for si in range(n_sites):
            n = site_counts[si]
            avg_within_var = site_vars[si].mean()
            residual_norm_sq = np.sum(residuals[si] ** 2)

            if residual_norm_sq < 1e-12:
                shrinkage_alpha[si] = 0.0
            else:
                shrink_factor = (logit_dim * avg_within_var) / (n * residual_norm_sq)
                shrinkage_alpha[si] = max(0.0, 1.0 - shrink_factor)

            shrunk_residuals[si] = shrinkage_alpha[si] * residuals[si]

        logger.info(
            "  shrinkage alpha: min=%.3f (n=%d), max=%.3f (n=%d)",
            shrinkage_alpha.min(), site_counts[shrinkage_alpha.argmin()],
            shrinkage_alpha.max(), site_counts[shrinkage_alpha.argmax()],
        )

        # === Mahalanobis-weighted residuals ===
        logger.info("SiteResiduals: computing Mahalanobis-weighted residuals")
        mahal_residuals = self._mahalanobis_weight(residuals)

        # === Quantile estimation (per site, from partitioned data) ===
        logger.info("SiteResiduals: quantile estimation")

        n_quantiles = len(QUANTILES)
        quantile_profiles = np.zeros(
            (n_sites, logit_dim, n_quantiles), dtype=np.float32
        )

        for si, site in enumerate(sites):
            df = load_site(site, self.features_dir)
            logits = logits_to_numpy(df)
            n = len(df)

            # Subsample if large
            if n > MAX_QUANTILE_SAMPLES:
                rng = np.random.RandomState(42)
                idx = rng.choice(n, MAX_QUANTILE_SAMPLES, replace=False)
                logits = logits[idx]

            for qi, q in enumerate(QUANTILES):
                quantile_profiles[si, :, qi] = np.quantile(logits, q, axis=0)

            del df, logits
            logger.info("  site %s: %d windows", site, n)

        # === Derived metrics ===
        iqr = quantile_profiles[:, :, 3] - quantile_profiles[:, :, 1]
        median_abs = np.abs(quantile_profiles[:, :, 2]) + 1e-6
        stability = 1.0 / (1.0 + iqr / median_abs)

        # === Save ===
        np.savez_compressed(
            self.output()[0].path,
            sites=np.array(sites),
            site_counts=site_counts,
            global_mean=global_mean.astype(np.float32),
            global_var=global_var.astype(np.float32),
            site_means=site_means.astype(np.float32),
            site_vars=site_vars.astype(np.float32),
            residuals=residuals.astype(np.float32),
            shrinkage_alpha=shrinkage_alpha.astype(np.float32),
            shrunk_residuals=shrunk_residuals.astype(np.float32),
            mahal_residuals=mahal_residuals.astype(np.float32),
            quantile_profiles=quantile_profiles,
            quantile_levels=np.array(QUANTILES),
            iqr=iqr.astype(np.float32),
            stability=stability.astype(np.float32),
        )
        logger.info("SiteResiduals: saved arrays to %s", self.output()[0].path)

        summary = self._build_summary(
            sites, site_counts, residuals, shrunk_residuals,
            mahal_residuals, shrinkage_alpha, quantile_profiles,
            iqr, stability, perch_labels,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SiteResiduals: saved summary to %s", self.output()[1].path)

    def _mahalanobis_weight(self, residuals: np.ndarray) -> np.ndarray:
        cov_path = Path(self.artifacts_dir) / "logit_covariance.npz"
        if not cov_path.exists():
            logger.warning("SiteResiduals: logit_covariance.npz not found, "
                           "falling back to unweighted residuals")
            return residuals.copy()

        logger.info("  loading logit covariance for Mahalanobis weighting")
        data = np.load(str(cov_path))
        cov = data["cov"].astype(np.float64)

        PCA_K = 500
        logger.info("  computing top-%d eigendecomposition of covariance", PCA_K)
        eigenvalues, eigenvectors = linalg.eigh(
            cov, subset_by_index=[cov.shape[0] - PCA_K, cov.shape[0] - 1]
        )
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        inv_eigenvalues = 1.0 / (eigenvalues + MAHAL_REG)

        projected = residuals @ eigenvectors
        weighted = projected * inv_eigenvalues[None, :]
        mahal = weighted @ eigenvectors.T

        logger.info(
            "  eigenvalue range: [%.4f, %.4f], condition number: %.1f",
            eigenvalues[-1], eigenvalues[0], eigenvalues[0] / max(eigenvalues[-1], 1e-10),
        )

        return mahal

    def _build_summary(
        self, sites, site_counts, residuals, shrunk_residuals,
        mahal_residuals, shrinkage_alpha, quantile_profiles,
        iqr, stability, perch_labels,
    ) -> dict:
        n_sites = len(sites)

        residual_var = np.var(shrunk_residuals, axis=0)
        top_disc_idx = np.argsort(-residual_var)[:TOP_K_SUMMARY]

        summary = {
            "n_sites": n_sites,
            "n_logits": residuals.shape[1],
            "quantile_levels": list(QUANTILES),
            "most_discriminative_logits": [],
            "sites": {},
        }

        for j in top_disc_idx:
            name = perch_labels[j] if j < len(perch_labels) else f"logit_{j}"
            summary["most_discriminative_logits"].append({
                "logit_index": int(j),
                "perch_species": name,
                "residual_variance": round(float(residual_var[j]), 6),
                "shrunk_residual_range": [
                    round(float(shrunk_residuals[:, j].min()), 4),
                    round(float(shrunk_residuals[:, j].max()), 4),
                ],
            })

        for si, site in enumerate(sites):
            n = int(site_counts[si])
            alpha = float(shrinkage_alpha[si])

            abs_resid = np.abs(shrunk_residuals[si])
            site_top_idx = np.argsort(-abs_resid)[:20]

            site_top = []
            for j in site_top_idx:
                name = perch_labels[j] if j < len(perch_labels) else f"logit_{j}"
                site_top.append({
                    "logit_index": int(j),
                    "perch_species": name,
                    "raw_residual": round(float(residuals[si, j]), 4),
                    "shrunk_residual": round(float(shrunk_residuals[si, j]), 4),
                    "mahal_residual": round(float(mahal_residuals[si, j]), 4),
                    "iqr": round(float(iqr[si, j]), 4),
                    "stability": round(float(stability[si, j]), 4),
                })

            median_vals = quantile_profiles[si, :, 2]
            stable_mask = (median_vals > 0.5) & (stability[si] > 0.5)
            n_stable = int(stable_mask.sum())
            n_wide_iqr = int((iqr[si] > 1.0).sum())

            summary["sites"][site] = {
                "n_windows": n,
                "shrinkage_alpha": round(alpha, 4),
                "n_stable_logits": n_stable,
                "n_wide_iqr_logits": n_wide_iqr,
                "mean_iqr": round(float(iqr[si].mean()), 4),
                "top_distinctive": site_top,
            }

        return summary

    def _load_perch_labels(self) -> list[str]:
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
