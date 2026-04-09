"""Zero-shot baseline: direct Perch logit readout.

For the 203 BirdCLEF species that have a direct match in Perch's 14,795-class
taxonomy, the corresponding logit is a free predictor — no training needed.
For 6 species with genus-level proxy matches, we take the max logit across
all Perch species in the same genus.

This establishes the floor: how much signal does Perch already contain for
BirdCLEF species without any learning?
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    load_labeled,
    load_species_mapping,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class ZeroShotBaseline(AnalysisTask):
    """Evaluate Perch logits as zero-shot predictors for BirdCLEF species."""

    def output(self):
        return self.artifact("zero_shot_metrics.json")

    def run(self):
        self.ensure_parent(self.output())

        mapping = load_species_mapping(self.artifacts_dir)
        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}

        df = load_labeled(self.features_dir)
        logits = logits_to_numpy(df)  # (739, 14795)
        Y = build_binary_label_matrix(df, species_list)  # (739, 234)

        results = {}

        # Direct-mapped species (203)
        for sp, perch_idx in mapping["direct_map"].items():
            sp_idx = species_to_idx.get(sp)
            if sp_idx is None:
                continue
            y_true = Y[:, sp_idx]
            if y_true.sum() < 2 or y_true.sum() == len(y_true):
                results[sp] = {"auroc": None, "ap": None, "n_pos": int(y_true.sum()), "method": "direct"}
                continue
            y_score = logits[:, perch_idx]
            results[sp] = {
                "auroc": float(roc_auc_score(y_true, y_score)),
                "ap": float(average_precision_score(y_true, y_score)),
                "n_pos": int(y_true.sum()),
                "method": "direct",
            }

        # Genus-proxy species (6) — max logit across proxy indices
        for sp, perch_indices in mapping["genus_proxy_map"].items():
            sp_idx = species_to_idx.get(sp)
            if sp_idx is None:
                continue
            y_true = Y[:, sp_idx]
            if y_true.sum() < 2 or y_true.sum() == len(y_true):
                results[sp] = {"auroc": None, "ap": None, "n_pos": int(y_true.sum()), "method": "genus_proxy"}
                continue
            y_score = logits[:, perch_indices].max(axis=1)
            results[sp] = {
                "auroc": float(roc_auc_score(y_true, y_score)),
                "ap": float(average_precision_score(y_true, y_score)),
                "n_pos": int(y_true.sum()),
                "method": "genus_proxy",
            }

        # Aggregate metrics
        aurocs = [v["auroc"] for v in results.values() if v["auroc"] is not None]
        aps = [v["ap"] for v in results.values() if v["ap"] is not None]

        summary = {
            "per_species": results,
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "macro_ap": float(np.mean(aps)) if aps else None,
            "n_evaluated": len(aurocs),
            "n_direct": sum(1 for v in results.values() if v["method"] == "direct"),
            "n_genus_proxy": sum(1 for v in results.values() if v["method"] == "genus_proxy"),
            "n_skipped": sum(1 for v in results.values() if v["auroc"] is None),
            "n_unmapped": len(mapping["unmapped"]),
        }

        with open(self.output().path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "ZeroShot: macro AUROC=%.3f, macro AP=%.3f (%d species evaluated, %d unmapped)",
            summary["macro_auroc"] or 0,
            summary["macro_ap"] or 0,
            summary["n_evaluated"],
            summary["n_unmapped"],
        )
