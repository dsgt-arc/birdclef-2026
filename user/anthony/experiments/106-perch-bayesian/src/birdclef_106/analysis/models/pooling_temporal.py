"""Temporal pooling: aggregate per-window predictions within 60s clips.

Soundscapes are 60s (12 × 5s windows). This task evaluates how different
aggregation strategies over the 12 windows affect clip-level performance.

Uses OOF predictions from the ridge task:
- Focal track: direct application of focal-trained W (inherently unbiased)
- Soundscape track: LOSO out-of-fold predictions

Reports both tracks for comparison. This is the simple temporal baseline
that any state-space model (UKF, DMD) must beat.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.models.ridge_empirical_bayes import EmpiricalBayesRidge
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    load_labeled,
)

logger = logging.getLogger(__name__)


def _pool_and_evaluate(scores, Y, filenames, species_list):
    """Pool per-window scores to clip level and evaluate."""
    n_species = len(species_list)

    # Group by clip
    clips = {}
    for i, fname in enumerate(filenames):
        if fname not in clips:
            clips[fname] = {"indices": [], "labels": Y[i].copy()}
        clips[fname]["indices"].append(i)
        clips[fname]["labels"] = np.maximum(clips[fname]["labels"], Y[i])

    n_clips = len(clips)

    results = {}
    for strategy_name in ["max", "mean", "softmax"]:
        clip_scores = np.zeros((n_clips, n_species))
        clip_labels = np.zeros((n_clips, n_species))

        for ci, (fname, clip) in enumerate(clips.items()):
            idx = clip["indices"]
            ws = scores[idx]
            if strategy_name == "max":
                clip_scores[ci] = ws.max(axis=0)
            elif strategy_name == "mean":
                clip_scores[ci] = ws.mean(axis=0)
            elif strategy_name == "softmax":
                weights = softmax(ws, axis=0)
                clip_scores[ci] = (weights * ws).sum(axis=0)
            clip_labels[ci] = clip["labels"]

        aurocs, aps = [], []
        for j in range(n_species):
            y = clip_labels[:, j]
            if y.sum() < 2 or y.sum() >= n_clips - 1:
                continue
            aurocs.append(roc_auc_score(y, clip_scores[:, j]))
            aps.append(average_precision_score(y, clip_scores[:, j]))

        results[strategy_name] = {
            "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
            "macro_ap": float(np.mean(aps)) if aps else 0.0,
            "n_evaluated": len(aurocs),
        }

    # Window-level metrics
    aurocs_w, aps_w = [], []
    for j in range(n_species):
        y = Y[:, j]
        if y.sum() < 2 or y.sum() >= len(y) - 1:
            continue
        aurocs_w.append(roc_auc_score(y, scores[:, j]))
        aps_w.append(average_precision_score(y, scores[:, j]))

    results["window_level"] = {
        "macro_auroc": float(np.mean(aurocs_w)) if aurocs_w else 0.0,
        "macro_ap": float(np.mean(aps_w)) if aps_w else 0.0,
        "n_evaluated": len(aurocs_w),
    }

    return results, n_clips


class TemporalPooling(AnalysisTask):
    """Evaluate temporal pooling on both focal and soundscape OOF predictions."""

    def requires(self):
        return EmpiricalBayesRidge(**self._params())

    def output(self):
        return self.artifact("temporal_pooling_metrics.json")

    def run(self):
        self.ensure_parent(self.output())

        species_list = build_species_list(self.artifacts_dir)
        df = load_labeled(self.features_dir)
        Y = build_binary_label_matrix(df, species_list).astype(np.float64)
        filenames = df["filename"].to_list()

        oof_data = np.load(self.input()[2].path)

        all_results = {}

        # Evaluate focal track OOF
        if "focal_oof" in oof_data:
            focal_scores = oof_data["focal_oof"].astype(np.float64)
            focal_config = str(oof_data.get("focal_config", "focal"))
            focal_results, n_clips = _pool_and_evaluate(focal_scores, Y, filenames, species_list)
            all_results["focal"] = {
                "config": focal_config,
                "strategies": focal_results,
            }
            for name, m in focal_results.items():
                logger.info("TemporalPooling [focal] %s: AUROC=%.3f, AP=%.3f",
                           name, m["macro_auroc"], m["macro_ap"])

        # Evaluate soundscape LOSO track OOF
        if "sc_oof" in oof_data:
            sc_scores = oof_data["sc_oof"].astype(np.float64)
            sc_config = str(oof_data.get("sc_config", "soundscape"))
            sc_results, n_clips = _pool_and_evaluate(sc_scores, Y, filenames, species_list)
            all_results["soundscape"] = {
                "config": sc_config,
                "strategies": sc_results,
            }
            for name, m in sc_results.items():
                logger.info("TemporalPooling [soundscape] %s: AUROC=%.3f, AP=%.3f",
                           name, m["macro_auroc"], m["macro_ap"])

        output = {
            "tracks": all_results,
            "n_clips": n_clips,
            "n_windows": len(df),
        }

        with open(self.output().path, "w") as f:
            json.dump(output, f, indent=2)
