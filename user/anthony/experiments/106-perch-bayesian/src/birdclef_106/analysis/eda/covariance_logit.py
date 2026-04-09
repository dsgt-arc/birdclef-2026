"""Logit covariance: 14795×14795 covariance of Perch logits from unlabeled soundscapes.

Each of the 14,795 Perch logit dimensions corresponds to a specific species in
Perch's pretrained taxonomy. The covariance matrix reveals which Perch species
co-activate in the Pantanal soundscapes — informative about acoustic communities
and useful as a structured prior for downstream regression.

The covariance is computed incrementally over 16 shards (~8K rows each) to keep
peak memory at ~2.2 GB (1.75 GB accumulator in float64 + ~450 MB per shard).

Uses: empirical Bayes prior for ridge regression (ridge.py),
      feature decorrelation for sparse logistic (sparse_logistic.py).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import load_unlabeled_shards, logits_to_numpy

logger = logging.getLogger(__name__)


class LogitCovariance(AnalysisTask):
    """Compute the 14795×14795 logit covariance from unlabeled soundscapes."""

    def output(self):
        return [
            self.artifact("logit_covariance.npz"),
            self.artifact("logit_covariance_derived.npz"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        # Incremental accumulation: maintain running sum and outer-product sum.
        # Avoids loading all 127K rows at once.
        sum_x = None
        sum_xx = None
        n_total = 0

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            X = logits_to_numpy(shard).astype(np.float64)
            n = X.shape[0]
            d = X.shape[1]

            if sum_x is None:
                sum_x = np.zeros(d, dtype=np.float64)
                sum_xx = np.zeros((d, d), dtype=np.float64)

            sum_x += X.sum(axis=0)
            sum_xx += X.T @ X
            n_total += n
            logger.info("LogitCovariance: shard %d, %d rows (total %d)", i, n, n_total)

        mean = sum_x / n_total
        cov = sum_xx / n_total - np.outer(mean, mean)

        # Regularize diagonal for numerical stability
        cov += 1e-6 * np.eye(cov.shape[0])

        np.savez_compressed(
            self.output()[0].path,
            mean=mean,  # float64 — preserve precision for downstream inversion
            cov=cov,    # float64 — 14795^2 * 8 bytes ≈ 1.75 GB uncompressed
            n_samples=n_total,
        )

        logger.info(
            "LogitCovariance: %d×%d matrix from %d samples", cov.shape[0], cov.shape[1], n_total,
        )

        # Cache derived quantities so downstream tasks don't recompute them.
        # Top-500 eigendecomposition (~2 min) + full inverse (~3 min).
        EIGEN_K = 500
        logger.info("LogitCovariance: computing top-%d eigendecomposition", EIGEN_K)
        eigenvalues, eigenvectors = linalg.eigh(
            cov, subset_by_index=[cov.shape[0] - EIGEN_K, cov.shape[0] - 1]
        )
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        logger.info("LogitCovariance: computing full inverse")
        cov_inv = linalg.inv(cov)

        np.savez_compressed(
            self.output()[1].path,
            eigenvalues=eigenvalues.astype(np.float64),
            eigenvectors=eigenvectors.astype(np.float32),  # float32 saves ~30 MB
            cov_inv=cov_inv,  # float64 for ridge precision
            eigen_k=EIGEN_K,
        )

        logger.info(
            "LogitCovariance: eigenvalue range [%.4f, %.4f], condition=%.1f, saved derived",
            eigenvalues[-1], eigenvalues[0], eigenvalues[0] / max(eigenvalues[-1], 1e-10),
        )
