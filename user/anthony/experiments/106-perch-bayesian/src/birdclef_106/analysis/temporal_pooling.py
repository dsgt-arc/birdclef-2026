"""Temporal pooling: aggregate per-window predictions within 60s clips.

Soundscapes are 60s (12 × 5s windows). This task evaluates how different
aggregation strategies over the 12 windows affect clip-level performance.

Uses LOSO out-of-fold predictions from the ridge task to avoid train-on-train
evaluation bias. Each window's score comes from a ridge model that was trained
without that window's site.

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
    load_labeled,
)

logger = logging.getLogger(__name__)


class TemporalPooling(AnalysisTask):
    """Evaluate max/mean/softmax pooling of ridge OOF predictions across clip windows."""

    def requires(self):
        return EmpiricalBayesRidge(**self._params())

    def output(self):
        return self.artifact("temporal_pooling_metrics.json")

    def run(self):
        self.ensure_parent(self.output())

        species_list = build_species_list(self.artifacts_dir)
        df = load_labeled(self.features_dir)
        Y = build_binary_label_matrix(df, species_list).astype(np.float64)

        # Load LOSO out-of-fold predictions (not train-on-train)
        oof_path = self.input()[2].path
        oof_data = np.load(oof_path)
        scores = oof_data["oof_predictions"].astype(np.float64)  # (739, 234)

        # Group by clip (filename)
        filenames = df["filename"].to_list()
        clips = {}
        for i, fname in enumerate(filenames):
            if fname not in clips:
                clips[fname] = {"indices": [], "labels": Y[i].copy()}
            clips[fname]["indices"].append(i)
            clips[fname]["labels"] = np.maximum(clips[fname]["labels"], Y[i])

        n_clips = len(clips)
        n_species = len(species_list)

        strategies = {}
        for strategy_name in ["max", "mean", "softmax"]:
            clip_scores = np.zeros((n_clips, n_species))
            clip_labels = np.zeros((n_clips, n_species))

            for ci, (fname, clip) in enumerate(clips.items()):
                idx = clip["indices"]
                window_scores = scores[idx]

                if strategy_name == "max":
                    clip_scores[ci] = window_scores.max(axis=0)
                elif strategy_name == "mean":
                    clip_scores[ci] = window_scores.mean(axis=0)
                elif strategy_name == "softmax":
                    weights = softmax(window_scores, axis=0)
                    clip_scores[ci] = (weights * window_scores).sum(axis=0)

                clip_labels[ci] = clip["labels"]

            aurocs, aps = [], []
            for j in range(n_species):
                y = clip_labels[:, j]
                if y.sum() < 2 or y.sum() >= n_clips - 1:
                    continue
                aurocs.append(roc_auc_score(y, clip_scores[:, j]))
                aps.append(average_precision_score(y, clip_scores[:, j]))

            strategies[strategy_name] = {
                "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
                "macro_ap": float(np.mean(aps)) if aps else 0.0,
                "n_evaluated": len(aurocs),
            }

        # Window-level OOF metrics for comparison
        aurocs_w, aps_w = [], []
        for j in range(n_species):
            y = Y[:, j]
            if y.sum() < 2 or y.sum() >= len(y) - 1:
                continue
            aurocs_w.append(roc_auc_score(y, scores[:, j]))
            aps_w.append(average_precision_score(y, scores[:, j]))

        strategies["window_level_oof"] = {
            "macro_auroc": float(np.mean(aurocs_w)) if aurocs_w else 0.0,
            "macro_ap": float(np.mean(aps_w)) if aps_w else 0.0,
            "n_evaluated": len(aurocs_w),
        }

        results = {
            "strategies": strategies,
            "n_clips": n_clips,
            "n_windows": len(df),
            "evaluation": "LOSO out-of-fold predictions (no train-on-train bias)",
        }

        with open(self.output().path, "w") as f:
            json.dump(results, f, indent=2)

        for name, m in strategies.items():
            logger.info("TemporalPooling %s: AUROC=%.3f, AP=%.3f", name, m["macro_auroc"], m["macro_ap"])
