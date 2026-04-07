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

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import load_unlabeled_shards, logits_to_numpy

logger = logging.getLogger(__name__)


class LogitCovariance(AnalysisTask):
    """Compute the 14795×14795 logit covariance from unlabeled soundscapes."""

    def output(self):
        return self.artifact("logit_covariance.npz")

    def run(self):
        self.ensure_parent(self.output())

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
            self.output().path,
            mean=mean.astype(np.float32),
            cov=cov.astype(np.float32),
            n_samples=n_total,
        )

        logger.info(
            "LogitCovariance: %d×%d matrix from %d samples, saved to %s",
            cov.shape[0], cov.shape[1], n_total, self.output().path,
        )
