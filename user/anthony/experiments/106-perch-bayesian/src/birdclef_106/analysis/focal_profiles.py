"""Focal species profiles: per-species Perch logit signatures from focal recordings.

For each of the 206 focal species, compute the mean Perch logit activation
pattern across all focal recordings of that species. This gives a sparse
signature vector (14,795-d) representing "what Perch sees when species X calls."

Sparsity helps: most logits are near-zero for any given species, only a handful
fire consistently. Background noise and unrelated activations average out across
hundreds of recordings, leaving a clean signature.

Classification: match soundscape windows against signatures via cosine similarity.
This is prototype classification in logit space, trained on focal data (265K windows)
— combining the strengths of focal (lots of clean labeled data) and logit space
(interpretable, sparse, species-level).

Also computes secondary-label co-occurrence profiles: which other species are
present when species X is the primary label? This captures ecological co-occurrence
structure from focal metadata.
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
    extract_site,
    load_focal_shards,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class FocalSpeciesProfiles(AnalysisTask):
    """Build per-species logit signatures from focal data, evaluate on soundscapes."""

    def output(self):
        return [
            self.artifact("focal_profiles.npz"),
            self.artifact("focal_profiles_metrics.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        n_species = len(species_list)

        # === Pass 1: accumulate per-species logit sums from focal data ===

        logger.info("FocalProfiles: accumulating per-species logit profiles from focal data")
        logit_dim = None
        species_sum = None    # (n_species, logit_dim)
        species_count = np.zeros(n_species, dtype=np.int64)

        # Also track secondary label co-occurrence
        cooccurrence = np.zeros((n_species, n_species), dtype=np.int64)

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard)  # (N, 14795)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                species_sum = np.zeros((n_species, logit_dim), dtype=np.float64)

            primary_labels = shard["primary_label"].to_list()
            ground_truths = shard["ground_truth"].to_list()

            for j in range(len(shard)):
                pl = str(primary_labels[j]).strip()
                si = species_to_idx.get(pl)
                if si is None:
                    continue
                species_sum[si] += logits[j].astype(np.float64)
                species_count[si] += 1

                # Co-occurrence from secondary labels
                gt = str(ground_truths[j]).strip()
                if gt:
                    for sp in gt.split(";"):
                        sp = sp.strip()
                        sj = species_to_idx.get(sp)
                        if sj is not None and sj != si:
                            cooccurrence[si, sj] += 1

            if i % 4 == 0:
                logger.info("  focal shard %d done", i)

        # Compute mean profiles (species with 0 recordings get zero vectors)
        mask = species_count > 0
        profiles = np.zeros_like(species_sum)
        profiles[mask] = species_sum[mask] / species_count[mask, None]

        n_with_data = int(mask.sum())
        logger.info("FocalProfiles: %d/%d species have focal data", n_with_data, n_species)

        # Sparsity analysis: how many logits are "active" per species?
        # Threshold: logit > mean + 2*std across all species profiles
        global_mean = profiles[mask].mean()
        global_std = profiles[mask].std()
        active_threshold = global_mean + 2 * global_std
        n_active_per_species = (profiles > active_threshold).sum(axis=1)

        # === Pass 2: evaluate on labeled soundscapes ===

        logger.info("FocalProfiles: evaluating on labeled soundscapes")
        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)
        sc_sites = np.array([extract_site(f) for f in sc_df["filename"].to_list()])
        unique_sites = sorted(set(sc_sites))

        # Cosine similarity: each soundscape window vs all species profiles
        # Only use species that have focal data
        active_species = [i for i in range(n_species) if species_count[i] > 0]
        profiles_active = profiles[active_species]  # (n_active, 14795)

        # Normalize profiles for cosine sim
        norms = np.linalg.norm(profiles_active, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        profiles_normed = profiles_active / norms

        # Score all soundscape windows
        sc_norms = np.linalg.norm(sc_logits, axis=1, keepdims=True)
        sc_norms = np.maximum(sc_norms, 1e-8)
        sc_normed = sc_logits / sc_norms

        sim_matrix = sc_normed @ profiles_normed.T  # (739, n_active)

        # Evaluate per species
        per_species = {}
        for idx_in_active, si in enumerate(active_species):
            sp = species_list[si]
            y_true = sc_Y[:, si]
            n_pos = int(y_true.sum())
            if n_pos < 2:
                per_species[sp] = {
                    "auroc": None, "ap": None, "n_pos": n_pos,
                    "n_focal_windows": int(species_count[si]),
                    "n_active_logits": int(n_active_per_species[si]),
                }
                continue

            y_score = sim_matrix[:, idx_in_active]
            per_species[sp] = {
                "auroc": float(roc_auc_score(y_true, y_score)),
                "ap": float(average_precision_score(y_true, y_score)),
                "n_pos": n_pos,
                "n_focal_windows": int(species_count[si]),
                "n_active_logits": int(n_active_per_species[si]),
            }

        # Top co-occurring species pairs
        top_cooc = []
        for si in range(n_species):
            for sj in range(si + 1, n_species):
                c = int(cooccurrence[si, sj] + cooccurrence[sj, si])
                if c > 10:
                    top_cooc.append({
                        "species_a": species_list[si],
                        "species_b": species_list[sj],
                        "count": c,
                    })
        top_cooc.sort(key=lambda x: -x["count"])
        top_cooc = top_cooc[:50]

        # Aggregate metrics
        aurocs = [v["auroc"] for v in per_species.values() if v["auroc"] is not None]
        aps = [v["ap"] for v in per_species.values() if v["ap"] is not None]

        summary = {
            "per_species": per_species,
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "macro_ap": float(np.mean(aps)) if aps else None,
            "n_evaluated": len(aurocs),
            "n_species_with_focal": n_with_data,
            "mean_active_logits": float(n_active_per_species[mask].mean()),
            "median_active_logits": float(np.median(n_active_per_species[mask])),
            "top_cooccurrences": top_cooc,
        }

        # Save profiles
        np.savez_compressed(
            self.output()[0].path,
            profiles=profiles.astype(np.float32),        # (234, 14795)
            species_count=species_count,                  # (234,)
            cooccurrence=cooccurrence.astype(np.int32),   # (234, 234)
            species_list=species_list,
            n_active_per_species=n_active_per_species,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "FocalProfiles: macro AUROC=%.3f, AP=%.3f (%d species evaluated, "
            "mean %.0f active logits/species)",
            summary["macro_auroc"] or 0, summary["macro_ap"] or 0,
            summary["n_evaluated"], summary["mean_active_logits"],
        )
