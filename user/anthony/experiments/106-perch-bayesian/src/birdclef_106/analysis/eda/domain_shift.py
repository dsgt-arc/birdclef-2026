"""Distribution shift analysis: focal vs soundscape Perch logit distributions.

Per-logit analysis is interpretable because each of the 14,795 dimensions
corresponds to a known Perch species. Identifying which logits shift most
between focal (controlled recordings) and soundscape (Pantanal field recordings)
reveals which species detections are unreliable across domains.

This is the go/no-go diagnostic: if features don't transfer, no downstream
model will work regardless of sophistication.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from scipy.stats import ks_2samp

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    load_focal_shards,
    load_labeled,
    load_unlabeled_shards,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class DistributionShift(AnalysisTask):
    """Compare Perch logit distributions between focal and soundscape data."""

    def output(self):
        return [
            self.artifact("distribution_shift.json"),
            self.artifact("distribution_shift.npz"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        # Load labeled soundscape logits (all 739)
        labeled_df = load_labeled(self.features_dir)
        labeled_logits = logits_to_numpy(labeled_df)

        # Sample ~10K from unlabeled soundscapes (first 2 shards ≈ 16K, subsample)
        unlabeled_parts = []
        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            unlabeled_parts.append(logits_to_numpy(shard))
            if i >= 1:
                break
        unlabeled_logits = np.concatenate(unlabeled_parts)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(unlabeled_logits), size=min(10000, len(unlabeled_logits)), replace=False)
        unlabeled_logits = unlabeled_logits[idx]

        # Sample ~10K from focal (first 2 shards, subsample)
        focal_parts = []
        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            focal_parts.append(logits_to_numpy(shard))
            if i >= 1:
                break
        focal_logits = np.concatenate(focal_parts)
        idx = rng.choice(len(focal_logits), size=min(10000, len(focal_logits)), replace=False)
        focal_logits = focal_logits[idx]

        # Combine soundscape = labeled + unlabeled
        soundscape_logits = np.concatenate([labeled_logits, unlabeled_logits])

        # Per-logit KS test (14795 tests)
        d = soundscape_logits.shape[1]
        ks_stats = np.zeros(d)
        ks_pvals = np.zeros(d)
        for j in range(d):
            stat, pval = ks_2samp(focal_logits[:, j], soundscape_logits[:, j])
            ks_stats[j] = stat
            ks_pvals[j] = pval

        # Bonferroni correction
        bonferroni_thresh = 0.01 / d
        n_significant = int((ks_pvals < bonferroni_thresh).sum())

        # Top 50 most shifted logits
        top_idx = np.argsort(-ks_stats)[:50]
        top_shifted = [
            {"logit_index": int(j), "ks_statistic": float(ks_stats[j]), "ks_pvalue": float(ks_pvals[j])}
            for j in top_idx
        ]

        # Mean logit comparison
        focal_mean = focal_logits.mean(axis=0)
        soundscape_mean = soundscape_logits.mean(axis=0)
        mean_diff = np.abs(focal_mean - soundscape_mean)
        top_mean_diff_idx = np.argsort(-mean_diff)[:50]
        top_mean_diff = [
            {"logit_index": int(j), "focal_mean": float(focal_mean[j]),
             "soundscape_mean": float(soundscape_mean[j]), "abs_diff": float(mean_diff[j])}
            for j in top_mean_diff_idx
        ]

        results = {
            "n_focal": len(focal_logits),
            "n_soundscape": len(soundscape_logits),
            "n_logit_dims": d,
            "n_significant_ks": n_significant,
            "frac_significant": float(n_significant / d),
            "median_ks_statistic": float(np.median(ks_stats)),
            "mean_ks_statistic": float(np.mean(ks_stats)),
            "top_shifted_logits": top_shifted,
            "top_mean_diff_logits": top_mean_diff,
        }

        with open(self.output()[0].path, "w") as f:
            json.dump(results, f, indent=2)

        # Save full per-dimension vectors for domain transfer weights
        np.savez_compressed(
            self.output()[1].path,
            ks_stats=ks_stats.astype(np.float32),
            ks_pvals=ks_pvals.astype(np.float32),
            focal_mean=focal_mean.astype(np.float32),
            soundscape_mean=soundscape_mean.astype(np.float32),
            mean_diff=mean_diff.astype(np.float32),
        )

        logger.info(
            "DistributionShift: %d/%d logits significantly shifted (%.1f%%)",
            n_significant, d, 100 * n_significant / d,
        )
