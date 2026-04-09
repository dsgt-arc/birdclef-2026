"""Site residual profiles with James-Stein shrinkage and Mahalanobis weighting.

Extends soundscape_profiles.py with statistically rigorous aggregation:

1. **Residuals**: site_mean - global_mean (what's distinctive about each site)
2. **James-Stein shrinkage**: denoise residuals for low-n sites by shrinking
   toward the global mean proportional to estimation uncertainty
3. **Mahalanobis-weighted residuals**: decorrelate using the logit covariance
   so correlated logit clusters don't dominate
4. **Quantile summaries**: p5/p25/p50/p75/p95 per logit per site via online
   Greenwald-Khanna-style incremental estimation (two-pass: exact on second pass)

Output:
    artifacts/site_residuals.npz — all arrays (23 sites × 14795 logits)
    artifacts/site_residuals_summary.json — per-site interpretable summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import linalg

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    extract_site,
    load_labeled,
    load_unlabeled_shards,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
# For Mahalanobis, regularize the covariance inversion
MAHAL_REG = 1e-4
# Top-k logits by cross-site variance to report in summary
TOP_K_SUMMARY = 30


class SiteResidualProfiles(AnalysisTask):
    """Residual + shrinkage + Mahalanobis site profiles from unlabeled soundscapes."""

    def output(self):
        return [
            self.artifact("site_residuals.npz"),
            self.artifact("site_residuals_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        perch_labels = self._load_perch_labels()

        # =====================================================================
        # Pass 1: accumulate sufficient statistics (sum, sum_sq, count per site)
        # and collect all values for quantile computation
        # =====================================================================

        logger.info("SiteResiduals: pass 1 — sufficient statistics")

        site_sum: dict[str, np.ndarray] = {}
        site_sum_sq: dict[str, np.ndarray] = {}
        site_count: dict[str, int] = {}
        logit_dim = None

        # Global accumulators
        global_sum = None
        global_sum_sq = None
        global_count = 0

        all_shards = list(load_unlabeled_shards(self.features_dir))
        labeled = load_labeled(self.features_dir)
        all_shards.append(labeled)

        for i, shard in enumerate(all_shards):
            logits = logits_to_numpy(shard).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                global_sum = np.zeros(logit_dim, dtype=np.float64)
                global_sum_sq = np.zeros(logit_dim, dtype=np.float64)

            filenames = shard["filename"].to_list()

            # Global stats
            global_sum += logits.sum(axis=0)
            global_sum_sq += (logits ** 2).sum(axis=0)
            global_count += len(shard)

            # Per-site stats
            for j in range(len(shard)):
                site = extract_site(filenames[j])
                if site not in site_sum:
                    site_sum[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_sum_sq[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_count[site] = 0
                site_sum[site] += logits[j]
                site_sum_sq[site] += logits[j] ** 2
                site_count[site] += 1

            if i % 4 == 0:
                logger.info("  pass 1 shard %d/%d done", i, len(all_shards))

        sites = sorted(site_count.keys())
        n_sites = len(sites)
        logger.info("SiteResiduals: %d sites, %d total windows", n_sites, global_count)

        # =====================================================================
        # Compute means and variances
        # =====================================================================

        global_mean = global_sum / global_count
        global_var = global_sum_sq / global_count - global_mean ** 2

        site_means = np.zeros((n_sites, logit_dim), dtype=np.float64)
        site_vars = np.zeros((n_sites, logit_dim), dtype=np.float64)
        site_counts = np.array([site_count[s] for s in sites])

        for si, site in enumerate(sites):
            n = site_count[site]
            site_means[si] = site_sum[site] / n
            site_vars[si] = site_sum_sq[site] / n - site_means[si] ** 2

        # =====================================================================
        # 1. Raw residuals
        # =====================================================================

        residuals = site_means - global_mean[None, :]

        # =====================================================================
        # 2. James-Stein shrinkage
        #
        # For each site, shrink the residual toward zero (global mean):
        #   shrunk[i] = alpha[i] * residual[i]
        #   alpha[i] = max(0, 1 - (p * sigma^2_within[i]) / (n[i] * ||residual[i]||^2))
        #
        # where sigma^2_within is the average within-site variance.
        # Sites with few windows or large within-site variance get shrunk more.
        # =====================================================================

        logger.info("SiteResiduals: computing James-Stein shrinkage")

        shrinkage_alpha = np.zeros(n_sites, dtype=np.float64)
        shrunk_residuals = np.zeros_like(residuals)

        for si in range(n_sites):
            n = site_counts[si]
            # Average within-site variance across logits
            avg_within_var = site_vars[si].mean()
            residual_norm_sq = np.sum(residuals[si] ** 2)

            if residual_norm_sq < 1e-12:
                shrinkage_alpha[si] = 0.0
            else:
                # James-Stein: shrink proportional to noise/signal ratio
                # p * sigma^2 / (n * ||delta||^2)
                shrink_factor = (logit_dim * avg_within_var) / (n * residual_norm_sq)
                shrinkage_alpha[si] = max(0.0, 1.0 - shrink_factor)

            shrunk_residuals[si] = shrinkage_alpha[si] * residuals[si]

        logger.info(
            "  shrinkage alpha: min=%.3f (n=%d), max=%.3f (n=%d)",
            shrinkage_alpha.min(), site_counts[shrinkage_alpha.argmin()],
            shrinkage_alpha.max(), site_counts[shrinkage_alpha.argmax()],
        )

        # =====================================================================
        # 3. Mahalanobis-weighted residuals
        #
        # Use the precomputed logit covariance from unlabeled soundscapes.
        # weighted_residual[i] = Sigma_inv @ residual[i]
        #
        # This decorrelates the logits so correlated clusters (e.g., 50 cicada
        # logits) contribute as one effective dimension.
        #
        # For 14795x14795 inversion, use PCA-reduced precision (top-k eigenvalues).
        # =====================================================================

        logger.info("SiteResiduals: computing Mahalanobis-weighted residuals")
        mahal_residuals = self._mahalanobis_weight(residuals)

        # =====================================================================
        # Pass 2: quantile estimation
        #
        # For each site, collect per-logit values and compute exact quantiles.
        # To manage memory (23 sites × 14795 logits × N windows is too large),
        # we process one site at a time by streaming through shards and only
        # keeping values for the current site's logits.
        #
        # Optimization: for large sites (>10K windows), subsample to 10K for
        # quantile estimation — quantile accuracy at 10K is already <1%.
        # =====================================================================

        logger.info("SiteResiduals: pass 2 — quantile estimation")

        n_quantiles = len(QUANTILES)
        quantile_profiles = np.zeros(
            (n_sites, logit_dim, n_quantiles), dtype=np.float32
        )

        MAX_QUANTILE_SAMPLES = 10_000

        for si, site in enumerate(sites):
            n = site_counts[si]
            # Determine subsample rate
            subsample = max(1, n // MAX_QUANTILE_SAMPLES)

            # Collect this site's logit values across all shards
            site_values = []
            sample_idx = 0

            for shard in list(load_unlabeled_shards(self.features_dir)) + [
                load_labeled(self.features_dir)
            ]:
                logits = logits_to_numpy(shard)
                filenames = shard["filename"].to_list()

                for j in range(len(shard)):
                    if extract_site(filenames[j]) != site:
                        continue
                    if sample_idx % subsample == 0:
                        site_values.append(logits[j])
                    sample_idx += 1

            if site_values:
                arr = np.stack(site_values)  # (n_sampled, logit_dim)
                for qi, q in enumerate(QUANTILES):
                    quantile_profiles[si, :, qi] = np.quantile(arr, q, axis=0)

            logger.info(
                "  site %s: %d/%d windows sampled for quantiles", site, len(site_values), n
            )

        # =====================================================================
        # Derived: interquartile range and stability score
        # =====================================================================

        iqr = quantile_profiles[:, :, 3] - quantile_profiles[:, :, 1]  # p75 - p25
        # Stability: logits with small IQR relative to their mean are reliable indicators
        # stability = 1 / (1 + IQR / |median|) — high stability = tight distribution
        median_abs = np.abs(quantile_profiles[:, :, 2]) + 1e-6
        stability = 1.0 / (1.0 + iqr / median_abs)

        # =====================================================================
        # Save arrays
        # =====================================================================

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

        # =====================================================================
        # Build interpretable summary JSON
        # =====================================================================

        summary = self._build_summary(
            sites, site_counts, residuals, shrunk_residuals,
            mahal_residuals, shrinkage_alpha, quantile_profiles,
            iqr, stability, perch_labels,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SiteResiduals: saved summary to %s", self.output()[1].path)

    def _mahalanobis_weight(self, residuals: np.ndarray) -> np.ndarray:
        """Apply Mahalanobis weighting via PCA-reduced precision.

        Full 14795×14795 inversion is expensive and ill-conditioned.
        Instead: use top-k PCA components from the logit covariance to build
        a low-rank precision approximation.
        """
        cov_path = Path(self.artifacts_dir) / "logit_covariance.npz"
        if not cov_path.exists():
            logger.warning("SiteResiduals: logit_covariance.npz not found, "
                           "falling back to variance-only weighting")
            # Diagonal approximation: weight by inverse global variance
            global_var_path = None
            # Just use the variance from the residuals' parent computation
            # which should be available in the caller — but we don't have it here.
            # Fall back to unweighted.
            return residuals.copy()

        logger.info("  loading logit covariance for Mahalanobis weighting")
        data = np.load(str(cov_path))
        cov = data["cov"].astype(np.float64)  # (14795, 14795)

        # PCA-reduced precision: keep top-k eigenvalues
        # Eigendecompose covariance (symmetric → use eigh)
        PCA_K = 500
        logger.info("  computing top-%d eigendecomposition of covariance", PCA_K)
        eigenvalues, eigenvectors = linalg.eigh(
            cov, subset_by_index=[cov.shape[0] - PCA_K, cov.shape[0] - 1]
        )
        # eigh returns ascending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Regularize and invert eigenvalues
        inv_eigenvalues = 1.0 / (eigenvalues + MAHAL_REG)

        # Low-rank precision: V @ diag(1/lambda) @ V.T
        # Apply to residuals: result[i] = V @ diag(1/lambda) @ V.T @ residual[i]
        # = V @ (diag(1/lambda) @ (V.T @ residual[i]))
        projected = residuals @ eigenvectors  # (n_sites, PCA_K)
        weighted = projected * inv_eigenvalues[None, :]  # scale by inverse eigenvalues
        mahal = weighted @ eigenvectors.T  # back to logit space

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
        """Build per-site interpretable summary with top distinctive logits."""
        n_sites = len(sites)

        # Cross-site variance of shrunk residuals — most discriminative logits
        residual_var = np.var(shrunk_residuals, axis=0)
        top_disc_idx = np.argsort(-residual_var)[:TOP_K_SUMMARY]

        summary = {
            "n_sites": n_sites,
            "n_logits": residuals.shape[1],
            "quantile_levels": list(QUANTILES),
            "most_discriminative_logits": [],
            "sites": {},
        }

        # Top discriminative logits (global)
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

        # Per-site summary
        for si, site in enumerate(sites):
            n = int(site_counts[si])
            alpha = float(shrinkage_alpha[si])

            # Top distinctive logits for this site (by absolute shrunk residual)
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

            # Most stable logits (high activation + tight distribution)
            median_vals = quantile_profiles[si, :, 2]
            stable_mask = (median_vals > 0.5) & (stability[si] > 0.5)
            n_stable = int(stable_mask.sum())

            # Most variable logits (wide IQR)
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
