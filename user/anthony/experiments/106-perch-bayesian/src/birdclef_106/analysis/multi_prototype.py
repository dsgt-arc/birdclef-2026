"""Multi-prototype species profiles via Leiden clustering on within-species residuals.

Instead of a single mean profile per species, cluster focal windows within each
species to find distinct call types (song, alarm, contact, etc.) and build one
prototype per cluster.

Pipeline:
    1. Pass 1: accumulate per-species means from focal shards.
    2. Pass 2: subsample within-species residuals for SVD whitening.
    3. Pass 3: re-read shards, project residuals into whitened 200-d space,
       store per-species projected vectors + original residuals for prototype
       reconstruction. Only 200-d projected vectors are stored in bulk; original
       residuals are accumulated per-species as running sums per cluster.
    4. Per-species Leiden clustering on dense Mahalanobis Gram matrix.
    5. Evaluate multi-prototype scoring on labeled soundscapes.

Memory optimization: stores projected vectors (200-d float32) not raw logits
(14795-d). Peak: ~266K × 200 × 4 = 200 MB for projected, plus per-species
raw residuals loaded one species at a time for prototype computation.

Outputs:
    multi_prototypes.npz — prototype matrix, species assignments, metadata
    multi_prototypes_summary.json — per-species cluster counts, sizes, validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.focal_profiles import FocalSpeciesProfiles
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    load_focal_shards,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

# Whitening PCA components
N_COMPONENTS = 200

# Minimum cluster size to become a prototype
MIN_CLUSTER_SIZE = 20

# Maximum prototypes per species
MAX_PROTOTYPES = 8

# Leiden resolution parameter
LEIDEN_RESOLUTION = 1.0

# Minimum mean pairwise cosine within a cluster to keep it
MIN_COHESION = 0.3


def _leiden_cluster(projected_normed, n_windows, ig, leidenalg):
    """Run Leiden on dense Mahalanobis Gram matrix. Returns labels array."""
    gram = projected_normed @ projected_normed.T
    np.fill_diagonal(gram, 0.0)
    gram = np.maximum(gram, 0.0)

    sources, targets = np.triu_indices(n_windows, k=1)
    weights = gram[sources, targets]
    pos_mask = weights > 0
    sources = sources[pos_mask]
    targets = targets[pos_mask]
    weights = weights[pos_mask]

    if len(sources) == 0:
        return None

    g = ig.Graph(n=n_windows,
                 edges=list(zip(sources.tolist(), targets.tolist())),
                 directed=False)
    g.es["weight"] = weights.tolist()

    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=LEIDEN_RESOLUTION, seed=42,
    )
    return np.array(partition.membership)


def _community_stats(labels, projected_normed):
    """Compute community statistics with cohesion and rejection reasons."""
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    all_communities = []
    for cl, count in zip(unique_labels, label_counts):
        cl_mask = labels == cl
        cl_proj = projected_normed[cl_mask]
        cl_gram = cl_proj @ cl_proj.T
        n_cl = cl_gram.shape[0]
        cohesion = (cl_gram.sum() - n_cl) / max(n_cl * (n_cl - 1), 1)

        if count < MIN_CLUSTER_SIZE:
            status = "rejected_too_small"
        elif cohesion < MIN_COHESION:
            status = "rejected_low_cohesion"
        else:
            status = "pending"

        all_communities.append({
            "id": int(cl), "size": int(count),
            "cohesion": round(float(cohesion), 4), "status": status,
        })

    # Accept in size order up to MAX_PROTOTYPES
    n_proto = 0
    accepted_ids = []
    for comm in sorted(all_communities, key=lambda c: -c["size"]):
        if comm["status"] != "pending":
            continue
        if n_proto >= MAX_PROTOTYPES:
            comm["status"] = "rejected_max_cap"
            continue
        comm["status"] = "accepted"
        accepted_ids.append(comm["id"])
        n_proto += 1

    sizes = [c["size"] for c in all_communities]
    cohesions = [c["cohesion"] for c in all_communities]

    stats = {
        "n_total": len(all_communities),
        "n_accepted": n_proto,
        "n_rejected_too_small": sum(1 for c in all_communities if c["status"] == "rejected_too_small"),
        "n_rejected_low_cohesion": sum(1 for c in all_communities if c["status"] == "rejected_low_cohesion"),
        "n_rejected_max_cap": sum(1 for c in all_communities if c["status"] == "rejected_max_cap"),
        "size_min": min(sizes), "size_max": max(sizes),
        "size_median": int(np.median(sizes)),
        "cohesion_min": round(min(cohesions), 4),
        "cohesion_max": round(max(cohesions), 4),
        "cohesion_median": round(float(np.median(cohesions)), 4),
    }

    return accepted_ids, stats, all_communities


class MultiPrototypeProfiles(AnalysisTask):
    """Build multi-prototype species profiles via Leiden clustering."""

    def requires(self):
        return FocalSpeciesProfiles(**self._params())

    def output(self):
        return [
            self.artifact("multi_prototypes.npz"),
            self.artifact("multi_prototypes_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        n_species = len(species_list)

        # =====================================================================
        # Pass 1: per-species means
        # =====================================================================

        logger.info("MultiPrototype: pass 1 — per-species means")

        logit_dim = None
        sp_sum = None
        sp_count = np.zeros(n_species, dtype=np.int64)

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                sp_sum = np.zeros((n_species, logit_dim), dtype=np.float64)

            pls = shard["primary_label"].to_list()
            shard_si = np.array(
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls],
                dtype=np.int32,
            )
            valid = shard_si >= 0
            if valid.any():
                valid_idx = np.where(valid)[0]
                np.add.at(sp_sum, shard_si[valid_idx], logits[valid_idx])
                np.add.at(sp_count, shard_si[valid_idx], 1)

            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        has_data = sp_count > 0
        sp_means = np.zeros_like(sp_sum)
        sp_means[has_data] = sp_sum[has_data] / sp_count[has_data, None]
        global_focal_mean = sp_sum[has_data].sum(axis=0) / sp_count[has_data].sum()

        logger.info("MultiPrototype: %d/%d species have focal data", has_data.sum(), n_species)

        # =====================================================================
        # Pass 2: subsample residuals for SVD (memory-light)
        # =====================================================================

        logger.info("MultiPrototype: pass 2 — subsampling residuals for SVD")

        total_focal = int(sp_count[has_data].sum())
        svd_subsample_rate = max(1, total_focal // 50_000)
        residual_samples = []

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            pls = shard["primary_label"].to_list()
            shard_si = np.array(
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls], dtype=np.int32,
            )
            valid = shard_si >= 0
            if valid.any():
                valid_idx = np.where(valid)[0]
                residuals_batch = logits[valid_idx] - sp_means[shard_si[valid_idx]]
                for j in range(0, len(valid_idx), svd_subsample_rate):
                    residual_samples.append(residuals_batch[j])
            if i % 4 == 0:
                logger.info("  pass 2 shard %d done (%d samples)", i, len(residual_samples))

        residual_matrix = np.array(residual_samples, dtype=np.float64)
        del residual_samples

        n_comp = min(N_COMPONENTS, residual_matrix.shape[0] - 1, residual_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(residual_matrix)
        components = svd.components_  # (n_comp, logit_dim)
        whiten_scale = 1.0 / np.maximum(svd.singular_values_, 1e-8)
        explained = svd.explained_variance_ratio_.sum()
        logger.info("MultiPrototype: %d components, %.1f%% variance", n_comp, explained * 100)
        del residual_matrix

        # =====================================================================
        # Pass 3: project residuals into whitened space (200-d) + store per-species
        #
        # We store the 200-d projected vectors (compact) plus per-species
        # cluster sums in full 14795-d for prototype reconstruction.
        # Peak memory: ~266K × 200 × 4 bytes = ~200 MB for projected.
        # =====================================================================

        logger.info("MultiPrototype: pass 3 — projecting to whitened space")

        # Pre-allocate per-species lists of projected vectors + raw residuals
        # We store projected (200-d) for clustering and raw residuals (14795-d)
        # for prototype centroid computation. However, raw residuals are too big
        # to store all at once. Instead, we'll cluster on projected, then re-read
        # shards to compute cluster centroids.
        #
        # Strategy: store projected + species_idx arrays. Cluster on projected.
        # Then pass 4 re-reads shards to compute centroids for accepted clusters.

        all_projected = []  # will be (total_focal, n_comp) float32
        all_species_idx = []  # species index per row

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            pls = shard["primary_label"].to_list()
            shard_si = np.array(
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls], dtype=np.int32,
            )
            valid = shard_si >= 0
            if valid.any():
                valid_idx = np.where(valid)[0]
                residuals = logits[valid_idx] - sp_means[shard_si[valid_idx]]
                projected = (residuals @ components.T) * whiten_scale[None, :]
                all_projected.append(projected.astype(np.float32))
                all_species_idx.append(shard_si[valid_idx])
            if i % 4 == 0:
                logger.info("  pass 3 shard %d done", i)

        all_projected = np.concatenate(all_projected)  # (total_focal, n_comp)
        all_species_idx = np.concatenate(all_species_idx)  # (total_focal,)
        logger.info("MultiPrototype: projected %d windows to %d-d", len(all_projected), n_comp)

        # =====================================================================
        # Cluster each species using projected vectors
        # =====================================================================

        logger.info("MultiPrototype: clustering")

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "Requires igraph and leidenalg. Install: uv pip install igraph leidenalg"
            )

        # Per-species: cluster labels (aligned with all_projected)
        # We'll store cluster assignments, then do one final pass to compute centroids.
        per_species_labels = {}  # si -> labels array
        per_species_info = {}
        per_species_accepted = {}  # si -> list of accepted cluster IDs

        active_species = sorted([si for si in range(n_species) if has_data[si]])

        for idx, si in enumerate(active_species):
            sp = species_list[si]
            mask = all_species_idx == si
            n_windows = int(mask.sum())

            if n_windows < MIN_CLUSTER_SIZE:
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                per_species_accepted[si] = None  # fallback to mean
                continue

            sp_projected = all_projected[mask]
            norms = np.maximum(np.linalg.norm(sp_projected, axis=1, keepdims=True), 1e-8)
            sp_normed = sp_projected / norms

            labels = _leiden_cluster(sp_normed, n_windows, ig, leidenalg)

            if labels is None:
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                per_species_accepted[si] = None
                continue

            accepted_ids, stats, communities = _community_stats(labels, sp_normed)

            if not accepted_ids:
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_cluster_passed_filters",
                    "community_stats": stats, "communities": communities,
                }
                per_species_accepted[si] = None
            else:
                per_species_info[sp] = {
                    "n_windows": n_windows,
                    "n_prototypes": len(accepted_ids),
                    "fallback": None,
                    "community_stats": stats,
                    "communities": communities,
                }
                per_species_labels[si] = labels
                per_species_accepted[si] = accepted_ids

            if idx % 20 == 0:
                n_p = len(accepted_ids) if accepted_ids else 1
                logger.info("  species %d/%d (%s): %d windows → %d prototypes",
                            idx, len(active_species), sp, n_windows, n_p)

        del all_projected  # free ~200 MB

        # =====================================================================
        # Pass 4: compute prototype centroids in original logit space
        #
        # Re-read shards, for each window check if it belongs to an accepted
        # cluster, accumulate into per-(species, cluster) sum.
        # =====================================================================

        logger.info("MultiPrototype: pass 4 — computing prototype centroids")

        # Initialize accumulators: (species_idx, cluster_id) -> (sum, count)
        cluster_sums = {}
        for si, accepted in per_species_accepted.items():
            if accepted is None:
                continue
            for cid in accepted:
                cluster_sums[(si, cid)] = [np.zeros(logit_dim, dtype=np.float64), 0]

        # We need to track which row in the per-species mask each window corresponds to
        # Rebuild per-species row counters
        sp_row_counters = np.zeros(n_species, dtype=np.int64)

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            pls = shard["primary_label"].to_list()

            for j in range(len(shard)):
                si = species_to_idx.get(str(pls[j]).strip(), -1)
                if si < 0:
                    continue
                row_in_species = int(sp_row_counters[si])
                sp_row_counters[si] += 1

                if si not in per_species_labels:
                    continue
                labels = per_species_labels[si]
                cid = int(labels[row_in_species])
                key = (si, cid)
                if key in cluster_sums:
                    residual = logits[j] - sp_means[si]
                    cluster_sums[key][0] += residual
                    cluster_sums[key][1] += 1

            if i % 4 == 0:
                logger.info("  pass 4 shard %d done", i)

        # Build prototype vectors
        all_prototypes = []
        proto_species_idx = []
        proto_cluster_id = []

        for si in active_species:
            sp = species_list[si]
            accepted = per_species_accepted.get(si)

            if accepted is None:
                # Fallback to mean
                prototype = (sp_means[si] - global_focal_mean).astype(np.float32)
                all_prototypes.append(prototype)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                continue

            for cid in accepted:
                s, c = cluster_sums[(si, cid)]
                if c > 0:
                    cluster_mean = s / c
                else:
                    cluster_mean = np.zeros(logit_dim)
                prototype = (sp_means[si] + cluster_mean - global_focal_mean).astype(np.float32)
                all_prototypes.append(prototype)
                proto_species_idx.append(si)
                proto_cluster_id.append(cid)

        # Stack
        prototype_matrix = np.stack(all_prototypes)
        proto_species_idx = np.array(proto_species_idx, dtype=np.int32)
        proto_cluster_id = np.array(proto_cluster_id, dtype=np.int32)

        proto_norms = np.maximum(
            np.linalg.norm(prototype_matrix, axis=1, keepdims=True), 1e-8
        )
        prototype_matrix_normed = prototype_matrix / proto_norms

        total_protos = len(all_prototypes)
        logger.info(
            "MultiPrototype: %d total prototypes across %d species (%.1f avg)",
            total_protos, len(per_species_info),
            total_protos / max(len(per_species_info), 1),
        )

        # =====================================================================
        # Evaluate on labeled soundscapes
        # =====================================================================

        logger.info("MultiPrototype: evaluating on labeled soundscapes")

        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)

        sc_centered = sc_logits - global_focal_mean[None, :]
        sc_norms = np.maximum(np.linalg.norm(sc_centered, axis=1, keepdims=True), 1e-8)
        sc_normed = sc_centered / sc_norms

        all_scores = sc_normed @ prototype_matrix_normed.T
        multi_scores = np.full((len(sc_df), n_species), -1.0)
        for pi in range(total_protos):
            si = proto_species_idx[pi]
            multi_scores[:, si] = np.maximum(multi_scores[:, si], all_scores[:, pi])

        single_means = sp_means.copy()
        single_means[has_data] -= global_focal_mean[None, :]
        single_norms = np.maximum(np.linalg.norm(single_means, axis=1, keepdims=True), 1e-8)
        single_normed = single_means / single_norms
        single_scores = sc_normed @ single_normed.T

        eval_results = {}
        for si in range(n_species):
            sp = species_list[si]
            if not has_data[si] or sc_Y[:, si].sum() < 2:
                continue
            y = sc_Y[:, si]
            auroc_m = float(roc_auc_score(y, multi_scores[:, si]))
            auroc_s = float(roc_auc_score(y, single_scores[:, si]))
            ap_m = float(average_precision_score(y, multi_scores[:, si]))
            ap_s = float(average_precision_score(y, single_scores[:, si]))
            eval_results[sp] = {
                "auroc_multi": round(auroc_m, 4), "auroc_single": round(auroc_s, 4),
                "auroc_delta": round(auroc_m - auroc_s, 4),
                "ap_multi": round(ap_m, 4), "ap_single": round(ap_s, 4),
                "ap_delta": round(ap_m - ap_s, 4),
            }

        aurocs_multi = [v["auroc_multi"] for v in eval_results.values()]
        aurocs_single = [v["auroc_single"] for v in eval_results.values()]
        aps_multi = [v["ap_multi"] for v in eval_results.values()]
        aps_single = [v["ap_single"] for v in eval_results.values()]
        n_improved = sum(1 for v in eval_results.values() if v["auroc_delta"] > 0)

        logger.info(
            "MultiPrototype eval: mean AUROC %.4f (multi) vs %.4f (single), %d/%d improved",
            np.mean(aurocs_multi), np.mean(aurocs_single), n_improved, len(eval_results),
        )

        # =====================================================================
        # Save
        # =====================================================================

        np.savez_compressed(
            self.output()[0].path,
            prototypes=prototype_matrix.astype(np.float32),
            prototypes_normed=prototype_matrix_normed.astype(np.float32),
            proto_species_idx=proto_species_idx,
            proto_cluster_id=proto_cluster_id,
            species_list=np.array(species_list),
            species_count=sp_count,
            global_focal_mean=global_focal_mean.astype(np.float32),
            whiten_components=components.astype(np.float32),
            whiten_scale=whiten_scale.astype(np.float32),
            explained_variance_ratio=svd.explained_variance_ratio_.astype(np.float32),
        )

        summary = {
            "n_species": n_species,
            "n_species_with_data": int(has_data.sum()),
            "total_prototypes": total_protos,
            "avg_prototypes_per_species": round(total_protos / max(len(per_species_info), 1), 2),
            "whiten_n_components": n_comp,
            "whiten_explained_variance": round(float(explained), 4),
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "leiden_resolution": LEIDEN_RESOLUTION,
            "eval": {
                "n_evaluated": len(eval_results),
                "mean_auroc_multi": round(float(np.mean(aurocs_multi)), 4) if aurocs_multi else None,
                "mean_auroc_single": round(float(np.mean(aurocs_single)), 4) if aurocs_single else None,
                "mean_ap_multi": round(float(np.mean(aps_multi)), 4) if aps_multi else None,
                "mean_ap_single": round(float(np.mean(aps_single)), 4) if aps_single else None,
                "n_improved_auroc": n_improved,
                "per_species": eval_results,
            },
            "per_species": per_species_info,
        }

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MultiPrototype: saved summary to %s", self.output()[1].path)
