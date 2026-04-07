"""Embedding covariance: 1536×1536 covariance of Perch embeddings from unlabeled soundscapes.

Complementary to the logit covariance (14795×14795). The embedding space is
Perch's penultimate acoustic representation — captures spectral/temporal features
before taxonomic projection.

What this reveals:
    - Dominant acoustic modes in the Pantanal soundscapes (PCA of this matrix)
    - Effective dimensionality of the embedding space for this domain
    - Better-conditioned prior for ridge regression (1536×1536 vs 14795×14795)

Much smaller than logit covariance: 1536² × 8 bytes = 19 MB vs 1.75 GB.
"""

from __future__ import annotations

import logging

import numpy as np

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import embeddings_to_numpy, load_unlabeled_shards

logger = logging.getLogger(__name__)


class EmbeddingCovariance(AnalysisTask):
    """Compute the 1536×1536 embedding covariance from unlabeled soundscapes."""

    def output(self):
        return self.artifact("embedding_covariance.npz")

    def run(self):
        self.ensure_parent(self.output())

        sum_x = None
        sum_xx = None
        n_total = 0

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            X = embeddings_to_numpy(shard).astype(np.float64)
            n = X.shape[0]
            d = X.shape[1]

            if sum_x is None:
                sum_x = np.zeros(d, dtype=np.float64)
                sum_xx = np.zeros((d, d), dtype=np.float64)

            sum_x += X.sum(axis=0)
            sum_xx += X.T @ X
            n_total += n
            logger.info("EmbeddingCovariance: shard %d, %d rows (total %d)", i, n, n_total)

        mean = sum_x / n_total
        cov = sum_xx / n_total - np.outer(mean, mean)
        cov += 1e-6 * np.eye(cov.shape[0])

        # Compute effective rank for diagnostics
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-8]
        p = eigvals / eigvals.sum()
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))

        np.savez_compressed(
            self.output().path,
            mean=mean,  # float64
            cov=cov,    # float64, ~19 MB
            n_samples=n_total,
            effective_rank=effective_rank,
        )

        logger.info(
            "EmbeddingCovariance: %d×%d matrix from %d samples, effective rank=%.0f, saved to %s",
            cov.shape[0], cov.shape[1], n_total, effective_rank, self.output().path,
        )
