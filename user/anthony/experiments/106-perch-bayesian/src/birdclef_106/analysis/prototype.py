"""Prototype classifier: nearest centroid in embedding space.

For the 25 unmapped insect sonotype species (47158sonXX) that have no Perch
logit mapping, we use embedding-space prototypes. Embeddings are preferred
over logits here because they preserve acoustic similarity for non-bird sounds
that Perch was never trained to classify.

Method: compute per-species embedding centroid from labeled windows, classify
new windows by cosine similarity to the nearest centroid. LOSO CV by site.

References:
    Snell et al. 2017 — Prototypical Networks for Few-Shot Learning
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    load_labeled,
    load_species_mapping,
)

logger = logging.getLogger(__name__)


class PrototypeClassifier(AnalysisTask):
    """Nearest-centroid classification for unmapped species in embedding space."""

    def output(self):
        return self.artifact("prototype_metrics.json")

    def run(self):
        self.ensure_parent(self.output())

        mapping = load_species_mapping(self.artifacts_dir)
        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        unmapped = mapping["unmapped"]

        df = load_labeled(self.features_dir)
        emb = embeddings_to_numpy(df)  # (739, 1536)
        Y = build_binary_label_matrix(df, species_list)  # (739, 234)
        sites = [extract_site(f) for f in df["filename"].to_list()]
        unique_sites = sorted(set(sites))
        site_arr = np.array(sites)

        results = {}
        for sp in unmapped:
            sp_idx = species_to_idx.get(sp)
            if sp_idx is None:
                continue
            y_true = Y[:, sp_idx]
            n_pos = int(y_true.sum())
            if n_pos < 2:
                results[sp] = {"auroc": None, "ap": None, "n_pos": n_pos, "note": "too few positives"}
                continue

            # LOSO CV
            y_scores = np.zeros(len(df), dtype=np.float64)
            for site in unique_sites:
                test_mask = site_arr == site
                train_mask = ~test_mask
                if y_true[train_mask].sum() == 0:
                    continue

                # Compute centroid from training positives
                pos_mask = train_mask & (y_true == 1)
                centroid = emb[pos_mask].mean(axis=0, keepdims=True)  # (1, 1536)

                # Cosine similarity as score
                y_scores[test_mask] = cosine_similarity(emb[test_mask], centroid).ravel()

            # Evaluate
            evaluated_mask = y_scores != 0
            if evaluated_mask.sum() < 2 or y_true[evaluated_mask].sum() < 1:
                results[sp] = {"auroc": None, "ap": None, "n_pos": n_pos, "note": "insufficient CV coverage"}
                continue

            results[sp] = {
                "auroc": float(roc_auc_score(y_true[evaluated_mask], y_scores[evaluated_mask])),
                "ap": float(average_precision_score(y_true[evaluated_mask], y_scores[evaluated_mask])),
                "n_pos": n_pos,
            }

        aurocs = [v["auroc"] for v in results.values() if v["auroc"] is not None]
        aps = [v["ap"] for v in results.values() if v["ap"] is not None]

        summary = {
            "per_species": results,
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "macro_ap": float(np.mean(aps)) if aps else None,
            "n_evaluated": len(aurocs),
            "n_unmapped_total": len(unmapped),
        }

        with open(self.output().path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Prototype: macro AUROC=%.3f, macro AP=%.3f (%d/%d species evaluated)",
            summary["macro_auroc"] or 0, summary["macro_ap"] or 0,
            summary["n_evaluated"], summary["n_unmapped_total"],
        )
