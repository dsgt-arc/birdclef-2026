"""Multi-prototype species profiles via Leiden clustering on within-species residuals.

Pipeline:
    1. Pass 1: per-species means from focal shards.
    2. Pass 2: subsample residuals → SVD whitening transform.
    3. Pass 3: write per-species .npz files to scratch (projected + residuals).
    4. Per species: load .npz, Leiden cluster, compute centroid prototypes.
    5. Evaluate on labeled soundscapes.

Per-species files on scratch (~15 GB total) make the pipeline robust to failure
and allow restarting from step 4 without re-reading all shards.

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
    """Compute per-community cohesion, apply size/cohesion/cap filters."""
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
        "n_total": len(communities),
        "n_accepted": len(accepted_ids),
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
        return FocalSpeciesProfiles(**self._params())

    def output(self):
        return [
            self.artifact("multi_prototypes.npz"),
            self.artifact("multi_prototypes_summary.json"),
        ]

    def _cache_dir(self):
        return Path(self.features_dir).expanduser() / "multi_proto_cache"

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        n_species = len(species_list)
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

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
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls], dtype=np.int32,
            )
            valid = shard_si >= 0
            if valid.any():
                np.add.at(sp_sum, shard_si[np.where(valid)[0]], logits[np.where(valid)[0]])
                np.add.at(sp_count, shard_si[np.where(valid)[0]], 1)
            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        has_data = sp_count > 0
        sp_means = np.zeros_like(sp_sum)
        sp_means[has_data] = sp_sum[has_data] / sp_count[has_data, None]
        global_focal_mean = sp_sum[has_data].sum(axis=0) / sp_count[has_data].sum()
        logger.info("MultiPrototype: %d/%d species have focal data", has_data.sum(), n_species)

        # =====================================================================
        # Pass 2: subsample residuals → SVD
        # =====================================================================

        logger.info("MultiPrototype: pass 2 — SVD")
        total_focal = int(sp_count[has_data].sum())
        svd_rate = max(1, total_focal // 50_000)
        residual_samples = []

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            pls = shard["primary_label"].to_list()
            shard_si = np.array(
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls], dtype=np.int32,
            )
            valid = shard_si >= 0
            if valid.any():
                vi = np.where(valid)[0]
                res = logits[vi] - sp_means[shard_si[vi]]
                for j in range(0, len(vi), svd_rate):
                    residual_samples.append(res[j])
            if i % 4 == 0:
                logger.info("  pass 2 shard %d (%d samples)", i, len(residual_samples))

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
        # Pass 3: write per-species .npz to scratch
        #
        # Each file: {projected: (N_s, 200) f32, residuals: (N_s, 14795) f32}
        # Total: ~15 GB on scratch. Enables restart from clustering step.
        # =====================================================================

        logger.info("MultiPrototype: pass 3 — writing per-species caches")

        # Collect per-species data as lists, flush after each shard
        sp_projected: dict[int, list] = {si: [] for si in range(n_species) if has_data[si]}
        sp_residuals: dict[int, list] = {si: [] for si in range(n_species) if has_data[si]}

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            pls = shard["primary_label"].to_list()
            shard_si = np.array(
                [species_to_idx.get(str(pl).strip(), -1) for pl in pls], dtype=np.int32,
            )
            valid = shard_si >= 0
            if valid.any():
                vi = np.where(valid)[0]
                vsi = shard_si[vi]
                res = logits[vi] - sp_means[vsi]
                proj = ((res @ components.T) * whiten_scale[None, :]).astype(np.float32)
                res_f32 = res.astype(np.float32)
                for j, si in enumerate(vsi):
                    sp_projected[si].append(proj[j])
                    sp_residuals[si].append(res_f32[j])
            if i % 4 == 0:
                logger.info("  pass 3 shard %d done", i)

        # Write per-species files
        for si in sorted(sp_projected.keys()):
            if not sp_projected[si]:
                continue
            out_path = cache_dir / f"species_{si:03d}.npz"
            np.savez(
                out_path,
                projected=np.stack(sp_projected[si]),
                residuals=np.stack(sp_residuals[si]),
            )
        del sp_projected, sp_residuals
        logger.info("MultiPrototype: wrote %d species caches to %s", sum(1 for si in range(n_species) if has_data[si]), cache_dir)

        # =====================================================================
        # Step 4: per-species clustering from cached files
        # =====================================================================

        logger.info("MultiPrototype: step 4 — clustering")

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError("Requires igraph and leidenalg.")

        all_prototypes = []
        proto_species_idx = []
        proto_cluster_id = []
        per_species_info = {}
        active = sorted([si for si in range(n_species) if has_data[si]])

        for idx, si in enumerate(active):
            sp = species_list[si]
            cache_path = cache_dir / f"species_{si:03d}.npz"

            if not cache_path.exists():
                # No data
                proto = (sp_means[si] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                per_species_info[sp] = {
                    "n_windows": 0, "n_prototypes": 1, "fallback": "no_cache",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                continue

            data = np.load(cache_path)
            projected = data["projected"]  # (N_s, 200)
            residuals = data["residuals"]  # (N_s, 14795)
            n_windows = projected.shape[0]

            if n_windows < MIN_CLUSTER_SIZE:
                proto = (sp_means[si] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1, "fallback": "too_few_windows",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                continue

            norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
            proj_normed = projected / norms

            labels = _leiden_cluster(proj_normed, n_windows, ig, leidenalg)

            if labels is None:
                proto = (sp_means[si] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1, "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                continue

            accepted_ids, stats, communities = _community_stats(labels, proj_normed)

            if not accepted_ids:
                proto = (sp_means[si] - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(0)
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_cluster_passed_filters",
                    "community_stats": stats,
                    "communities": sorted(communities, key=lambda c: -c["size"]),
                }
                continue

            # Build prototypes from accepted clusters
            for cid in accepted_ids:
                cl_residuals = residuals[labels == cid].astype(np.float64)
                cluster_mean = cl_residuals.mean(axis=0)
                proto = (sp_means[si] + cluster_mean - global_focal_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_species_idx.append(si)
                proto_cluster_id.append(cid)

            per_species_info[sp] = {
                "n_windows": n_windows, "n_prototypes": len(accepted_ids), "fallback": None,
                "community_stats": stats,
                "communities": sorted(communities, key=lambda c: -c["size"]),
            }

            if idx % 20 == 0:
                logger.info("  species %d/%d (%s): %d windows → %d prototypes",
                            idx, len(active), sp, n_windows, len(accepted_ids))

        # Stack prototypes
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
        # Evaluate on labeled soundscapes
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

        single_means = sp_means.copy()
        single_means[has_data] -= global_focal_mean[None, :]
        single_norms = np.maximum(np.linalg.norm(single_means, axis=1, keepdims=True), 1e-8)
        single_scores = sc_normed @ (single_means / single_norms).T

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
            "min_cohesion": MIN_COHESION,
            "cache_dir": str(cache_dir),
            "eval": {
                "n_evaluated": len(eval_results),
                "mean_auroc_multi": round(float(np.mean(aurocs_m)), 4) if aurocs_m else None,
                "mean_auroc_single": round(float(np.mean(aurocs_s)), 4) if aurocs_s else None,
                "n_improved_auroc": n_improved,
                "per_species": eval_results,
            },
            "per_species": per_species_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MultiPrototype: done")
