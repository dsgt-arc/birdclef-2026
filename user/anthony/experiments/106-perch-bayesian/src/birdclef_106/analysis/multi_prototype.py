"""Multi-prototype species profiles via Leiden clustering on within-species residuals.

Depends on PartitionFeatures which provides per-species parquets at
focal/by_species/{species}.parquet. Each species is processed independently:
load parquet → compute residuals → project → Leiden cluster → prototypes.

Pipeline:
    1. Load all species parquets to compute global mean + SVD whitening.
    2. Per species: load parquet, whiten residuals, dense Gram → Leiden,
       extract prototype centroids from accepted clusters.
    3. Evaluate on labeled soundscapes.

Memory: one species at a time (~5K × 14795 × 8 = ~600 MB peak).

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
from birdclef_106.analysis.partition import PartitionFeatures
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    list_focal_species,
    load_focal_species,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

N_COMPONENTS = 200
MIN_CLUSTER_SIZE = 20
MAX_PROTOTYPES = 8
LEIDEN_RESOLUTION = 1.0
MIN_COHESION = 0.3


def _leiden_cluster(projected_normed, n_windows, ig, leidenalg):
    """Run Leiden on dense Mahalanobis Gram matrix. Returns labels or None."""
    gram = projected_normed @ projected_normed.T
    np.fill_diagonal(gram, 0.0)
    gram = np.maximum(gram, 0.0)
    sources, targets = np.triu_indices(n_windows, k=1)
    weights = gram[sources, targets]
    pos_mask = weights > 0
    if pos_mask.sum() == 0:
        return None
    g = ig.Graph(
        n=n_windows,
        edges=list(zip(sources[pos_mask].tolist(), targets[pos_mask].tolist())),
        directed=False,
    )
    g.es["weight"] = weights[pos_mask].tolist()
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=LEIDEN_RESOLUTION, seed=42,
    )
    return np.array(partition.membership)


def _community_stats(labels, projected_normed):
    """Compute per-community cohesion, apply filters, return accepted IDs."""
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    communities = []
    for cl, count in zip(unique_labels, label_counts):
        cl_proj = projected_normed[labels == cl]
        cl_gram = cl_proj @ cl_proj.T
        n_cl = cl_gram.shape[0]
        cohesion = (cl_gram.sum() - n_cl) / max(n_cl * (n_cl - 1), 1)
        if count < MIN_CLUSTER_SIZE:
            status = "rejected_too_small"
        elif cohesion < MIN_COHESION:
            status = "rejected_low_cohesion"
        else:
            status = "pending"
        communities.append({
            "id": int(cl), "size": int(count),
            "cohesion": round(float(cohesion), 4), "status": status,
        })

    accepted_ids = []
    for comm in sorted(communities, key=lambda c: -c["size"]):
        if comm["status"] != "pending":
            continue
        if len(accepted_ids) >= MAX_PROTOTYPES:
            comm["status"] = "rejected_max_cap"
            continue
        comm["status"] = "accepted"
        accepted_ids.append(comm["id"])

    sizes = [c["size"] for c in communities]
    cohesions = [c["cohesion"] for c in communities]
    stats = {
        "n_total": len(communities), "n_accepted": len(accepted_ids),
        "n_rejected_too_small": sum(1 for c in communities if c["status"] == "rejected_too_small"),
        "n_rejected_low_cohesion": sum(1 for c in communities if c["status"] == "rejected_low_cohesion"),
        "n_rejected_max_cap": sum(1 for c in communities if c["status"] == "rejected_max_cap"),
        "size_min": min(sizes), "size_max": max(sizes), "size_median": int(np.median(sizes)),
        "cohesion_min": round(min(cohesions), 4), "cohesion_max": round(max(cohesions), 4),
        "cohesion_median": round(float(np.median(cohesions)), 4),
    }
    return accepted_ids, stats, communities


class MultiPrototypeProfiles(AnalysisTask):
    """Build multi-prototype species profiles via Leiden clustering."""

    def requires(self):
        return PartitionFeatures(**self._params())

    def output(self):
        return [
            self.artifact("multi_prototypes.npz"),
            self.artifact("multi_prototypes_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        n_species = len(species_list)
        available = set(list_focal_species(self.features_dir))

        # =====================================================================
        # Step 1: compute per-species means + global mean + SVD whitening
        #
        # One pass per species parquet. Only means + SVD subsample in memory.
        # =====================================================================
        logger.info("MultiPrototype: step 1 — per-species means + SVD")

        logit_dim = None
        sp_means = {}
        sp_counts = {}
        global_sum = None
        global_count = 0
        residual_samples = []

        for sp in species_list:
            if sp not in available:
                continue
            df = load_focal_species(sp, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                global_sum = np.zeros(logit_dim, dtype=np.float64)

            sp_mean = logits.mean(axis=0)
            sp_means[sp] = sp_mean
            sp_counts[sp] = len(df)
            global_sum += logits.sum(axis=0)
            global_count += len(df)

            # Subsample residuals for SVD
            residuals = logits - sp_mean[None, :]
            step = max(1, len(df) // 250)  # ~250 samples per species, ~50K total
            for j in range(0, len(df), step):
                residual_samples.append(residuals[j])

        global_focal_mean = global_sum / global_count
        logger.info("MultiPrototype: %d species with data, %d total windows",
                     len(sp_means), global_count)

        # Fit SVD on subsampled residuals
        res_matrix = np.array(residual_samples, dtype=np.float64)
        del residual_samples
        n_comp = min(N_COMPONENTS, res_matrix.shape[0] - 1, res_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(res_matrix)
        components = svd.components_
        whiten_scale = 1.0 / np.maximum(svd.singular_values_, 1e-8)
        explained = svd.explained_variance_ratio_.sum()
        logger.info("MultiPrototype: %d components, %.1f%% variance", n_comp, explained * 100)
        del res_matrix

        # =====================================================================
        # Step 2: per-species clustering
        #
        # Load each species' parquet, compute residuals, project, Leiden,
        # extract prototypes. One species in memory at a time.
        # =====================================================================
        logger.info("MultiPrototype: step 2 — per-species clustering")

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError("Requires igraph and leidenalg.")

        all_prototypes = []
        proto_species_idx = []
        proto_cluster_id = []
        per_species_info = {}
        species_to_idx = {s: i for i, s in enumerate(species_list)}

        for idx, sp in enumerate(species_list):
            si = species_to_idx[sp]

            if sp not in sp_means:
                per_species_info[sp] = {
                    "n_windows": 0, "n_prototypes": 1, "fallback": "no_data",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                proto = (-global_focal_mean).astype(np.float32)  # zero-centered
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                continue

            n_windows = sp_counts[sp]
            if n_windows < MIN_CLUSTER_SIZE:
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                proto = (sp_means[sp] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                continue

            # Load this species' data
            df = load_focal_species(sp, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)
            residuals = logits - sp_means[sp][None, :]

            # Project + whiten
            projected = (residuals @ components.T) * whiten_scale[None, :]
            norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
            proj_normed = (projected / norms).astype(np.float32)

            # Leiden on dense Gram
            labels = _leiden_cluster(proj_normed, n_windows, ig, leidenalg)

            if labels is None:
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                proto = (sp_means[sp] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                continue

            accepted_ids, stats, communities = _community_stats(labels, proj_normed)

            if not accepted_ids:
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_cluster_passed_filters",
                    "community_stats": stats,
                    "communities": sorted(communities, key=lambda c: -c["size"]),
                }
                proto = (sp_means[sp] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
            else:
                for cid in accepted_ids:
                    cluster_mean = residuals[labels == cid].mean(axis=0)
                    proto = (sp_means[sp] + cluster_mean - global_focal_mean).astype(np.float32)
                    all_prototypes.append(proto)
                    proto_species_idx.append(si)
                    proto_cluster_id.append(cid)

                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": len(accepted_ids),
                    "fallback": None, "community_stats": stats,
                    "communities": sorted(communities, key=lambda c: -c["size"]),
                }

            if idx % 20 == 0:
                n_p = per_species_info[sp]["n_prototypes"]
                logger.info("  species %d/%d (%s): %d windows → %d prototypes",
                            idx, n_species, sp, n_windows, n_p)

        # Stack
        prototype_matrix = np.stack(all_prototypes)
        proto_species_idx = np.array(proto_species_idx, dtype=np.int32)
        proto_cluster_id = np.array(proto_cluster_id, dtype=np.int32)
        proto_norms = np.maximum(np.linalg.norm(prototype_matrix, axis=1, keepdims=True), 1e-8)
        prototype_matrix_normed = prototype_matrix / proto_norms
        total_protos = len(all_prototypes)

        logger.info("MultiPrototype: %d prototypes across %d species (%.1f avg)",
                     total_protos, len(per_species_info),
                     total_protos / max(len(per_species_info), 1))

        # =====================================================================
        # Step 3: evaluate
        # =====================================================================
        logger.info("MultiPrototype: evaluating")
        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)

        sc_centered = sc_logits - global_focal_mean[None, :]
        sc_norms = np.maximum(np.linalg.norm(sc_centered, axis=1, keepdims=True), 1e-8)
        sc_normed = sc_centered / sc_norms

        all_scores = sc_normed @ prototype_matrix_normed.T
        multi_scores = np.full((len(sc_df), n_species), -1.0)
        for pi in range(total_protos):
            multi_scores[:, proto_species_idx[pi]] = np.maximum(
                multi_scores[:, proto_species_idx[pi]], all_scores[:, pi])

        # Single-mean baseline
        has_data = np.array([sp in sp_means for sp in species_list])
        single_mat = np.zeros((n_species, logit_dim), dtype=np.float64)
        for i, sp in enumerate(species_list):
            if sp in sp_means:
                single_mat[i] = sp_means[sp] - global_focal_mean
        s_norms = np.maximum(np.linalg.norm(single_mat, axis=1, keepdims=True), 1e-8)
        single_scores = sc_normed @ (single_mat / s_norms).T

        eval_results = {}
        for si in range(n_species):
            sp = species_list[si]
            if not has_data[si] or sc_Y[:, si].sum() < 2:
                continue
            y = sc_Y[:, si]
            am = float(roc_auc_score(y, multi_scores[:, si]))
            a_s = float(roc_auc_score(y, single_scores[:, si]))
            pm = float(average_precision_score(y, multi_scores[:, si]))
            ps = float(average_precision_score(y, single_scores[:, si]))
            eval_results[sp] = {
                "auroc_multi": round(am, 4), "auroc_single": round(a_s, 4),
                "auroc_delta": round(am - a_s, 4),
                "ap_multi": round(pm, 4), "ap_single": round(ps, 4),
                "ap_delta": round(pm - ps, 4),
            }

        aurocs_m = [v["auroc_multi"] for v in eval_results.values()]
        aurocs_s = [v["auroc_single"] for v in eval_results.values()]
        n_improved = sum(1 for v in eval_results.values() if v["auroc_delta"] > 0)
        logger.info("MultiPrototype eval: AUROC %.4f (multi) vs %.4f (single), %d/%d improved",
                     np.mean(aurocs_m), np.mean(aurocs_s), n_improved, len(eval_results))

        # =====================================================================
        # Save
        # =====================================================================
        sp_count = np.array([sp_counts.get(sp, 0) for sp in species_list], dtype=np.int64)
        np.savez_compressed(
            self.output()[0].path,
            prototypes=prototype_matrix.astype(np.float32),
            prototypes_normed=prototype_matrix_normed.astype(np.float32),
            proto_species_idx=proto_species_idx, proto_cluster_id=proto_cluster_id,
            species_list=np.array(species_list), species_count=sp_count,
            global_focal_mean=global_focal_mean.astype(np.float32),
            whiten_components=components.astype(np.float32),
            whiten_scale=whiten_scale.astype(np.float32),
            explained_variance_ratio=svd.explained_variance_ratio_.astype(np.float32),
        )

        summary = {
            "n_species": n_species, "n_species_with_data": len(sp_means),
            "total_prototypes": total_protos,
            "avg_prototypes_per_species": round(total_protos / max(len(per_species_info), 1), 2),
            "whiten_n_components": n_comp,
            "whiten_explained_variance": round(float(explained), 4),
            "min_cluster_size": MIN_CLUSTER_SIZE, "leiden_resolution": LEIDEN_RESOLUTION,
            "min_cohesion": MIN_COHESION,
            "eval": {
                "n_evaluated": len(eval_results),
                "mean_auroc_multi": round(float(np.mean(aurocs_m)), 4) if aurocs_m else None,
                "mean_auroc_single": round(float(np.mean(aurocs_s)), 4) if aurocs_s else None,
                "n_improved_auroc": n_improved, "per_species": eval_results,
            },
            "per_species": per_species_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MultiPrototype: done")
