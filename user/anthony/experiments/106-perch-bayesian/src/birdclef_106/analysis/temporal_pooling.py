"""Temporal pooling: aggregate per-window predictions within 60s clips.

Soundscapes are 60s (12 × 5s windows). This task evaluates how different
aggregation strategies over the 12 windows affect clip-level performance.

This is the simple temporal baseline that any state-space model (UKF, DMD)
must beat to justify its complexity.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.ridge import EmpiricalBayesRidge
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class TemporalPooling(AnalysisTask):
    """Evaluate max/mean/softmax pooling of ridge predictions across clip windows."""

    def requires(self):
        return EmpiricalBayesRidge(**self._params())

    def output(self):
        return self.artifact("temporal_pooling_metrics.json")

    def run(self):
        self.ensure_parent(self.output())

        species_list = build_species_list(self.artifacts_dir)
        df = load_labeled(self.features_dir)

        # Load ridge model
        model_path = self.input()[0].path
        model = np.load(model_path, allow_pickle=True)
        W = model["W"]
        feature_space = str(model["feature_space"])

        if feature_space == "embeddings":
            X = embeddings_to_numpy(df).astype(np.float64)
        else:
            X = logits_to_numpy(df).astype(np.float64)

        Y = build_binary_label_matrix(df, species_list).astype(np.float64)

        # Per-window predictions
        scores = X @ W  # (739, 234)

        # Group by clip (filename)
        filenames = df["filename"].to_list()
        clips = {}
        for i, fname in enumerate(filenames):
            if fname not in clips:
                clips[fname] = {"indices": [], "labels": Y[i]}
            clips[fname]["indices"].append(i)
            # Union of labels across windows
            clips[fname]["labels"] = np.maximum(clips[fname]["labels"], Y[i])

        n_clips = len(clips)
        n_species = len(species_list)

        # Apply pooling strategies
        strategies = {}
        for strategy_name in ["max", "mean", "softmax"]:
            clip_scores = np.zeros((n_clips, n_species))
            clip_labels = np.zeros((n_clips, n_species))

            for ci, (fname, clip) in enumerate(clips.items()):
                idx = clip["indices"]
                window_scores = scores[idx]  # (n_windows, 234)

                if strategy_name == "max":
                    clip_scores[ci] = window_scores.max(axis=0)
                elif strategy_name == "mean":
                    clip_scores[ci] = window_scores.mean(axis=0)
                elif strategy_name == "softmax":
                    weights = softmax(window_scores, axis=0)
                    clip_scores[ci] = (weights * window_scores).sum(axis=0)

                clip_labels[ci] = clip["labels"]

            # Evaluate
            aurocs, aps = [], []
            for j in range(n_species):
                y = clip_labels[:, j]
                if y.sum() < 1 or y.sum() == n_clips:
                    continue
                aurocs.append(roc_auc_score(y, clip_scores[:, j]))
                aps.append(average_precision_score(y, clip_scores[:, j]))

            strategies[strategy_name] = {
                "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
                "macro_ap": float(np.mean(aps)) if aps else 0.0,
                "n_evaluated": len(aurocs),
            }

        # Also report window-level metrics for comparison
        aurocs_w, aps_w = [], []
        for j in range(n_species):
            y = Y[:, j]
            if y.sum() < 1 or y.sum() == len(y):
                continue
            aurocs_w.append(roc_auc_score(y, scores[:, j]))
            aps_w.append(average_precision_score(y, scores[:, j]))

        strategies["window_level"] = {
            "macro_auroc": float(np.mean(aurocs_w)) if aurocs_w else 0.0,
            "macro_ap": float(np.mean(aps_w)) if aps_w else 0.0,
            "n_evaluated": len(aurocs_w),
        }

        results = {
            "strategies": strategies,
            "n_clips": n_clips,
            "n_windows": len(df),
            "feature_space": feature_space,
        }

        with open(self.output().path, "w") as f:
            json.dump(results, f, indent=2)

        for name, m in strategies.items():
            logger.info("TemporalPooling %s: AUROC=%.3f, AP=%.3f", name, m["macro_auroc"], m["macro_ap"])
