"""Multi-prototype species profiles via Leiden clustering on within-species residuals.

Instead of a single mean profile per species, cluster focal windows within each
species to find distinct call types (song, alarm, contact, etc.) and build one
prototype per cluster. Silent/noisy windows are dropped as isolated graph nodes.

Pipeline:
    1. Two passes over focal shards: accumulate per-species means, then collect
       residuals (windows - species mean) for the whitening transform.
    2. SVD on pooled within-species residuals → whitening transform.
    3. Per species: project residuals into whitened space, build k-NN cosine graph,
       run Leiden community detection, extract prototype centroids.
    4. Evaluate multi-prototype scoring on labeled soundscapes vs single-mean baseline.

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

# Whitening PCA components (captures within-species variation structure)
N_COMPONENTS = 200

# Minimum cluster size to become a prototype
MIN_CLUSTER_SIZE = 20

# Maximum prototypes per species (cap for very fragmented species)
MAX_PROTOTYPES = 8

# Leiden resolution parameter (higher = more clusters)
LEIDEN_RESOLUTION = 1.0

# Minimum mean pairwise cosine within a cluster to keep it as a prototype.
# Low-cohesion clusters are grab-bags of dissimilar windows, not real call types.
MIN_COHESION = 0.3


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
        # Pass 1: per-species means and window counts
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
                valid_si = shard_si[valid_idx]
                # Vectorized scatter-add: accumulate logits by species index
                np.add.at(sp_sum, valid_si, logits[valid_idx])
                np.add.at(sp_count, valid_si, 1)

            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        has_data = sp_count > 0
        sp_means = np.zeros_like(sp_sum)
        sp_means[has_data] = sp_sum[has_data] / sp_count[has_data, None]

        logger.info(
            "MultiPrototype: %d/%d species have focal data", has_data.sum(), n_species
        )

        # =====================================================================
        # Pass 2: collect within-species residuals, fit whitening transform
        #
        # We subsample residuals for SVD fitting (memory), then do a full pass
        # to collect per-species windows for clustering.
        # =====================================================================

        logger.info("MultiPrototype: pass 2 — collecting residuals for SVD")

        # Subsample residuals for SVD (target ~50K rows)
        max_svd_samples = 50_000
        total_focal = int(sp_count[has_data].sum())
        svd_subsample_rate = max(1, total_focal // max_svd_samples)

        residual_samples = []
        species_windows: dict[int, list[np.ndarray]] = {
            si: [] for si in range(n_species) if has_data[si]
        }

        row_counter = 0
        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            pls = shard["primary_label"].to_list()

            # Map labels to species indices (vectorized lookup)
            shard_si = np.array(
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls],
                dtype=np.int32,
            )
            valid = shard_si >= 0

            if valid.any():
                # Vectorized residual computation: subtract each row's species mean
                valid_idx = np.where(valid)[0]
                valid_si = shard_si[valid_idx]
                residuals_batch = logits[valid_idx] - sp_means[valid_si]

                # Dispatch to per-species lists
                for j, si in enumerate(valid_si):
                    species_windows[si].append(residuals_batch[j].astype(np.float32))

                # Subsample for SVD
                for j in range(0, len(valid_idx), max(1, svd_subsample_rate)):
                    residual_samples.append(residuals_batch[j])

            if i % 4 == 0:
                logger.info("  pass 2 shard %d done", i)

        residual_matrix = np.array(residual_samples, dtype=np.float64)
        logger.info(
            "MultiPrototype: SVD on %d residual samples (%d dims)",
            residual_matrix.shape[0], residual_matrix.shape[1],
        )

        # =====================================================================
        # Fit whitening transform via truncated SVD on pooled residuals
        # =====================================================================

        n_comp = min(N_COMPONENTS, residual_matrix.shape[0] - 1, residual_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(residual_matrix)

        # Whitening: project and scale by 1/sqrt(eigenvalue)
        # eigenvalues = singular_values^2 / (n-1)
        singular_values = svd.singular_values_
        components = svd.components_  # (n_comp, logit_dim)
        # Scale factors for whitening: 1 / singular_value (proportional to 1/sqrt(eigenvalue))
        whiten_scale = 1.0 / np.maximum(singular_values, 1e-8)

        explained = svd.explained_variance_ratio_.sum()
        logger.info(
            "MultiPrototype: %d components explain %.1f%% of within-species variance",
            n_comp, explained * 100,
        )

        del residual_matrix, residual_samples

        # =====================================================================
        # Per-species: whiten residuals, build k-NN graph, Leiden clustering
        # =====================================================================

        logger.info("MultiPrototype: clustering %d species", len(species_windows))

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "Multi-prototype clustering requires igraph and leidenalg. "
                "Install with: uv pip install igraph leidenalg"
            )

        # Global focal mean for centering prototypes in compositions
        global_focal_mean = sp_sum[has_data].sum(axis=0) / sp_count[has_data].sum()

        all_prototypes = []  # list of (logit_dim,) arrays
        proto_species_idx = []  # which species each prototype belongs to
        proto_cluster_id = []  # cluster ID within species
        per_species_info = {}

        for si in sorted(species_windows.keys()):
            sp = species_list[si]
            residuals_list = species_windows[si]
            n_windows = len(residuals_list)

            if n_windows < MIN_CLUSTER_SIZE:
                # Too few windows — fall back to single mean
                prototype = sp_means[si] - global_focal_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                per_species_info[sp] = {
                    "n_windows": n_windows,
                    "n_kept": n_windows,
                    "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "clusters": [{"id": 0, "size": n_windows}],
                }
                continue

            residuals = np.stack(residuals_list).astype(np.float64)

            # Whiten: project into PCA space, scale by inverse singular values
            projected = (residuals @ components.T) * whiten_scale[None, :]

            # Normalize for cosine similarity (in whitened space = Mahalanobis)
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            projected_normed = projected / norms

            # Dense Mahalanobis-weighted similarity graph
            # Gram matrix: cosine in whitened space = Mahalanobis in P-space
            gram = projected_normed @ projected_normed.T  # (N_s, N_s)

            # Zero out negative similarities and diagonal for Leiden
            np.fill_diagonal(gram, 0.0)
            gram = np.maximum(gram, 0.0)

            # Build igraph from dense weighted adjacency (upper triangle)
            sources, targets = np.triu_indices(n_windows, k=1)
            weights = gram[sources, targets]
            # Only keep edges with positive weight
            pos_mask = weights > 0
            sources = sources[pos_mask]
            targets = targets[pos_mask]
            weights = weights[pos_mask]

            if len(sources) == 0:
                prototype = sp_means[si] - global_focal_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                per_species_info[sp] = {
                    "n_windows": n_windows,
                    "n_prototypes": 1,
                    "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0},
                    "communities": [],
                }
                continue

            g = ig.Graph(n=n_windows,
                         edges=list(zip(sources.tolist(), targets.tolist())),
                         directed=False)
            g.es["weight"] = weights.tolist()

            # Leiden clustering on dense weighted graph
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=LEIDEN_RESOLUTION,
                seed=42,
            )

            labels = np.array(partition.membership)
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            # Compute cohesion for ALL communities and track rejection reasons
            all_communities = []
            for cl, count in zip(unique_labels, label_counts):
                cl_mask = labels == cl
                cl_projected = projected_normed[cl_mask]

                # Cohesion: mean pairwise cosine within cluster
                gram = cl_projected @ cl_projected.T
                n_cl = gram.shape[0]
                cohesion = (gram.sum() - n_cl) / max(n_cl * (n_cl - 1), 1)

                if count < MIN_CLUSTER_SIZE:
                    status = "rejected_too_small"
                elif cohesion < MIN_COHESION:
                    status = "rejected_low_cohesion"
                else:
                    status = "pending"  # will be accepted or capped

                all_communities.append({
                    "id": int(cl),
                    "size": int(count),
                    "cohesion": round(float(cohesion), 4),
                    "status": status,
                })

            # Accept communities in size order, up to MAX_PROTOTYPES
            accepted = []
            n_proto = 0
            for comm in sorted(all_communities, key=lambda c: -c["size"]):
                if comm["status"] != "pending":
                    continue
                if n_proto >= MAX_PROTOTYPES:
                    comm["status"] = "rejected_max_cap"
                    continue
                comm["status"] = "accepted"
                accepted.append(comm)
                n_proto += 1

            # Build prototypes from accepted communities
            for comm in accepted:
                cl = comm["id"]
                cl_mask = labels == cl
                cluster_mean = residuals[cl_mask].mean(axis=0)
                prototype = (sp_means[si] + cluster_mean) - global_focal_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_species_idx.append(si)
                proto_cluster_id.append(cl)

            if n_proto == 0:
                prototype = sp_means[si] - global_focal_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                n_proto = 1
                fallback = "no_cluster_passed_filters"
            else:
                fallback = None

            # Community statistics
            n_rejected_small = sum(1 for c in all_communities if c["status"] == "rejected_too_small")
            n_rejected_cohesion = sum(1 for c in all_communities if c["status"] == "rejected_low_cohesion")
            n_rejected_cap = sum(1 for c in all_communities if c["status"] == "rejected_max_cap")
            sizes = [c["size"] for c in all_communities]
            cohesions = [c["cohesion"] for c in all_communities]

            per_species_info[sp] = {
                "n_windows": n_windows,
                "n_prototypes": n_proto,
                "fallback": fallback,
                "community_stats": {
                    "n_total": len(all_communities),
                    "n_accepted": n_proto if fallback is None else 0,
                    "n_rejected_too_small": n_rejected_small,
                    "n_rejected_low_cohesion": n_rejected_cohesion,
                    "n_rejected_max_cap": n_rejected_cap,
                    "size_min": min(sizes),
                    "size_max": max(sizes),
                    "size_median": int(np.median(sizes)),
                    "cohesion_min": round(min(cohesions), 4),
                    "cohesion_max": round(max(cohesions), 4),
                    "cohesion_median": round(float(np.median(cohesions)), 4),
                },
                "communities": sorted(all_communities, key=lambda c: -c["size"]),
            }

            if si % 20 == 0:
                logger.info(
                    "  species %d/%d (%s): %d windows → %d prototypes",
                    si, n_species, sp, n_windows, n_proto,
                )

        # Stack all prototypes
        prototype_matrix = np.stack(all_prototypes)  # (total_protos, logit_dim)
        proto_species_idx = np.array(proto_species_idx, dtype=np.int32)
        proto_cluster_id = np.array(proto_cluster_id, dtype=np.int32)

        # Normalize prototypes for cosine scoring
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
        # Evaluate on labeled soundscapes: multi-prototype vs single-mean
        # =====================================================================

        logger.info("MultiPrototype: evaluating on labeled soundscapes")

        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)

        # Center and normalize soundscape windows
        sc_centered = sc_logits - global_focal_mean[None, :]
        sc_norms = np.maximum(np.linalg.norm(sc_centered, axis=1, keepdims=True), 1e-8)
        sc_normed = sc_centered / sc_norms

        # Multi-prototype scoring: max over prototypes per species
        all_scores = sc_normed @ prototype_matrix_normed.T  # (739, total_protos)
        multi_scores = np.full((len(sc_df), n_species), -1.0)
        for pi in range(total_protos):
            si = proto_species_idx[pi]
            multi_scores[:, si] = np.maximum(multi_scores[:, si], all_scores[:, pi])

        # Single-mean scoring (baseline from FocalSpeciesProfiles)
        single_means = sp_means.copy()
        single_means[has_data] -= global_focal_mean[None, :]
        single_norms = np.maximum(
            np.linalg.norm(single_means, axis=1, keepdims=True), 1e-8
        )
        single_normed = single_means / single_norms
        single_scores = sc_normed @ single_normed.T  # (739, n_species)

        # Per-species AUROC comparison
        eval_results = {}
        for si in range(n_species):
            sp = species_list[si]
            if not has_data[si]:
                continue
            y = sc_Y[:, si]
            if y.sum() < 2:
                continue

            auroc_multi = float(roc_auc_score(y, multi_scores[:, si]))
            auroc_single = float(roc_auc_score(y, single_scores[:, si]))
            ap_multi = float(average_precision_score(y, multi_scores[:, si]))
            ap_single = float(average_precision_score(y, single_scores[:, si]))

            eval_results[sp] = {
                "auroc_multi": round(auroc_multi, 4),
                "auroc_single": round(auroc_single, 4),
                "auroc_delta": round(auroc_multi - auroc_single, 4),
                "ap_multi": round(ap_multi, 4),
                "ap_single": round(ap_single, 4),
                "ap_delta": round(ap_multi - ap_single, 4),
            }

        # Aggregate metrics
        aurocs_multi = [v["auroc_multi"] for v in eval_results.values()]
        aurocs_single = [v["auroc_single"] for v in eval_results.values()]
        aps_multi = [v["ap_multi"] for v in eval_results.values()]
        aps_single = [v["ap_single"] for v in eval_results.values()]
        n_improved = sum(1 for v in eval_results.values() if v["auroc_delta"] > 0)

        logger.info(
            "MultiPrototype eval: mean AUROC %.4f (multi) vs %.4f (single), "
            "%d/%d species improved",
            np.mean(aurocs_multi), np.mean(aurocs_single),
            n_improved, len(eval_results),
        )

        # =====================================================================
        # Save outputs
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
        logger.info("MultiPrototype: saved arrays to %s", self.output()[0].path)

        summary = {
            "n_species": n_species,
            "n_species_with_data": int(has_data.sum()),
            "total_prototypes": total_protos,
            "avg_prototypes_per_species": round(
                total_protos / max(len(per_species_info), 1), 2
            ),
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
