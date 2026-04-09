"""Species co-occurrence graph via window-level scoring.

For each focal window, scores it against every species' mean signature.
Aggregates per species pair to build an asymmetric (k×k) similarity matrix.

Two representations (logit + embedding), plus a bipartite structure showing
which Perch taxa mediate each connection.

**Window-level scoring** (vs mean-to-mean):
    - Each window of species A is scored against species B's mean profile
    - A->B = mean similarity of A's windows to B's mean
    - Asymmetric: A->B != B->A (captures directional acoustic similarity)
    - Naturally handles within-species variance without explicit covariance

Two passes through focal shards:
    Pass 1: accumulate per-species means (sum + count)
    Pass 2: score each window against all species means, accumulate (k×k)

Outputs:
    species_graph.npz — asymmetric similarity matrices + species lists
    species_graph_summary.json — top neighbors per species, mediating Perch taxa
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.profiles_focal import FocalSpeciesProfiles
from birdclef_106.data_loading import (
    build_species_list,
    embeddings_to_numpy,
    load_focal_shards,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

TOP_K_NEIGHBORS = 10
TOP_K_MEDIATORS = 5


class SpeciesGraph(AnalysisTask):
    """Species co-occurrence graph from window-level focal scoring."""

    def requires(self):
        return FocalSpeciesProfiles(**self._params())

    def output(self):
        return [
            self.artifact("species_graph.npz"),
            self.artifact("species_graph_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        n_species = len(species_list)
        perch_labels = self._load_perch_labels()

        # =================================================================
        # Pass 1: accumulate per-species mean profiles
        # =================================================================

        logger.info("SpeciesGraph: pass 1 — per-species means")

        logit_dim = None
        emb_dim = None
        sp_logit_sum = None
        sp_emb_sum = None
        sp_count = np.zeros(n_species, dtype=np.int64)

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            embs = embeddings_to_numpy(shard).astype(np.float64)

            if logit_dim is None:
                logit_dim = logits.shape[1]
                emb_dim = embs.shape[1]
                sp_logit_sum = np.zeros((n_species, logit_dim), dtype=np.float64)
                sp_emb_sum = np.zeros((n_species, emb_dim), dtype=np.float64)

            pls = shard["primary_label"].to_list()
            for j in range(len(shard)):
                si = species_to_idx.get(str(pls[j]).strip())
                if si is None:
                    continue
                sp_logit_sum[si] += logits[j]
                sp_emb_sum[si] += embs[j]
                sp_count[si] += 1

            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        has_data = sp_count > 0
        n_with_data = int(has_data.sum())
        logger.info("SpeciesGraph: %d/%d species have focal data", n_with_data, n_species)

        # Compute mean profiles
        logit_means = np.zeros_like(sp_logit_sum)
        emb_means = np.zeros_like(sp_emb_sum)
        logit_means[has_data] = sp_logit_sum[has_data] / sp_count[has_data, None]
        emb_means[has_data] = sp_emb_sum[has_data] / sp_count[has_data, None]

        # Global mean across all focal windows (for centering)
        global_logit_mean = sp_logit_sum[has_data].sum(axis=0) / sp_count[has_data].sum()
        global_emb_mean = sp_emb_sum[has_data].sum(axis=0) / sp_count[has_data].sum()

        # Center: subtract global mean so cosine = Pearson correlation
        # This removes the shared background (e.g., ubiquitous insect logits)
        logit_centered = logit_means.copy()
        emb_centered = emb_means.copy()
        logit_centered[has_data] -= global_logit_mean[None, :]
        emb_centered[has_data] -= global_emb_mean[None, :]

        # Normalize centered profiles
        logit_norms = np.maximum(
            np.linalg.norm(logit_centered, axis=1, keepdims=True), 1e-8
        )
        emb_norms = np.maximum(
            np.linalg.norm(emb_centered, axis=1, keepdims=True), 1e-8
        )
        logit_means_normed = logit_centered / logit_norms  # (k, n)
        emb_means_normed = emb_centered / emb_norms  # (k, d)

        # =================================================================
        # Pass 2: window-level scoring
        #
        # For each window of species i, compute cosine similarity to every
        # species' mean. Accumulate into (k, k) matrices.
        #
        # cooc_logit[i, j] += cosine(window, logit_mean[j])
        # cooc_emb[i, j]   += cosine(window, emb_mean[j])
        #
        # Vectorized: per shard, compute scores = normed_windows @ means.T
        # then scatter-add by species index.
        # =================================================================

        logger.info("SpeciesGraph: pass 2 — window-level scoring")

        cooc_logit = np.zeros((n_species, n_species), dtype=np.float64)
        cooc_emb = np.zeros((n_species, n_species), dtype=np.float64)
        window_count = np.zeros(n_species, dtype=np.int64)

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            embs = embeddings_to_numpy(shard).astype(np.float64)

            # Center and normalize windows (same global mean as profiles)
            logits -= global_logit_mean[None, :]
            embs -= global_emb_mean[None, :]
            l_norms = np.maximum(np.linalg.norm(logits, axis=1, keepdims=True), 1e-8)
            e_norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8)
            logits_normed = logits / l_norms
            embs_normed = embs / e_norms

            # Score every window against every species mean: (batch, k)
            logit_scores = logits_normed @ logit_means_normed.T
            emb_scores = embs_normed @ emb_means_normed.T

            # Accumulate by species
            pls = shard["primary_label"].to_list()
            for j in range(len(shard)):
                si = species_to_idx.get(str(pls[j]).strip())
                if si is None:
                    continue
                cooc_logit[si] += logit_scores[j]
                cooc_emb[si] += emb_scores[j]
                window_count[si] += 1

            if i % 4 == 0:
                logger.info("  pass 2 shard %d done", i)

        # Normalize: average similarity per species pair
        # cooc[i,j] = mean cosine of species i's windows to species j's mean
        for si in range(n_species):
            if window_count[si] > 0:
                cooc_logit[si] /= window_count[si]
                cooc_emb[si] /= window_count[si]

        # Zero out species without data
        no_data = ~has_data
        cooc_logit[no_data, :] = 0
        cooc_logit[:, no_data] = 0
        cooc_emb[no_data, :] = 0
        cooc_emb[:, no_data] = 0

        logger.info(
            "SpeciesGraph: logit cooc range [%.4f, %.4f], emb cooc range [%.4f, %.4f]",
            cooc_logit[has_data][:, has_data].min(),
            cooc_logit[has_data][:, has_data].max(),
            cooc_emb[has_data][:, has_data].min(),
            cooc_emb[has_data][:, has_data].max(),
        )

        # Asymmetry: how different is A->B from B->A?
        asym_logit = cooc_logit - cooc_logit.T
        asym_emb = cooc_emb - cooc_emb.T

        # =================================================================
        # Save arrays
        # =================================================================

        np.savez_compressed(
            self.output()[0].path,
            species_list=np.array(species_list),
            species_count=sp_count,
            logit_cooc=cooc_logit.astype(np.float32),
            emb_cooc=cooc_emb.astype(np.float32),
            logit_asymmetry=asym_logit.astype(np.float32),
            emb_asymmetry=asym_emb.astype(np.float32),
            logit_means_normed=logit_means_normed.astype(np.float32),
            emb_means_normed=emb_means_normed.astype(np.float32),
            global_logit_mean=global_logit_mean.astype(np.float32),
            global_emb_mean=global_emb_mean.astype(np.float32),
        )
        logger.info("SpeciesGraph: saved arrays to %s", self.output()[0].path)

        # =================================================================
        # Build interpretable summary
        # =================================================================

        summary = self._build_summary(
            species_list, sp_count, has_data,
            cooc_emb, cooc_logit, logit_means_normed,
            asym_logit, asym_emb, perch_labels,
        )

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SpeciesGraph: saved summary to %s", self.output()[1].path)

    def _build_summary(
        self, species_list, sp_count, has_data,
        cooc_emb, cooc_logit, logit_means_normed,
        asym_logit, asym_emb, perch_labels,
    ) -> dict:
        n_species = len(species_list)
        active_idx = [i for i in range(n_species) if has_data[i]]

        summary = {
            "n_species": n_species,
            "n_with_data": len(active_idx),
            "species": {},
        }

        for si in active_idx:
            sp = species_list[si]

            # Top neighbors by embedding window-level similarity (exclude self)
            emb_scores = cooc_emb[si].copy()
            emb_scores[si] = -1
            emb_top_idx = np.argsort(-emb_scores)[:TOP_K_NEIGHBORS]

            emb_neighbors = []
            for ni in emb_top_idx:
                if not has_data[ni]:
                    continue
                emb_neighbors.append({
                    "species": species_list[ni],
                    "similarity": round(float(cooc_emb[si, ni]), 4),
                    "reverse": round(float(cooc_emb[ni, si]), 4),
                    "asymmetry": round(float(asym_emb[si, ni]), 4),
                })

            # Top neighbors by logit window-level similarity
            logit_scores = cooc_logit[si].copy()
            logit_scores[si] = -1
            logit_top_idx = np.argsort(-logit_scores)[:TOP_K_NEIGHBORS]

            logit_neighbors = []
            for ni in logit_top_idx:
                if not has_data[ni]:
                    continue

                # Mediating Perch species: element-wise product of normalized means
                joint = logit_means_normed[si] * logit_means_normed[ni]
                mediator_idx = np.argsort(-joint)[:TOP_K_MEDIATORS]
                mediators = []
                for mi in mediator_idx:
                    if joint[mi] < 1e-6:
                        break
                    name = perch_labels[mi] if mi < len(perch_labels) else f"logit_{mi}"
                    mediators.append({
                        "logit_index": int(mi),
                        "perch_species": name,
                        "joint_score": round(float(joint[mi]), 6),
                    })

                logit_neighbors.append({
                    "species": species_list[ni],
                    "similarity": round(float(cooc_logit[si, ni]), 4),
                    "reverse": round(float(cooc_logit[ni, si]), 4),
                    "asymmetry": round(float(asym_logit[si, ni]), 4),
                    "mediating_perch_taxa": mediators,
                })

            summary["species"][sp] = {
                "n_focal_windows": int(sp_count[si]),
                "emb_neighbors": emb_neighbors,
                "logit_neighbors": logit_neighbors,
            }

        return summary

    def _load_perch_labels(self) -> list[str]:
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
