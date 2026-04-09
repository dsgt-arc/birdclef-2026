"""Site-state prototypes via Leiden clustering on within-site residuals.

Depends on PartitionFeatures which provides per-site parquets at
soundscape_unlabeled/by_site/{site}.parquet. Each site is processed independently.

Pipeline:
    1. Load all site parquets to compute global mean + SVD whitening.
    2. Per site: load parquet, whiten residuals, dense Gram → Leiden,
       extract prototype centroids. Diel characterization from hours.
    3. Compute species × site-state affinity.

Memory: one site at a time (~40K × 14795 × 8 = ~4.7 GB peak for S22).

Outputs:
    site_prototypes.npz — prototype matrix, site assignments, whitening transform
    site_prototypes_summary.json — per-site cluster counts, temporal profiles, top taxa
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.partition import PartitionFeatures
from birdclef_106.data_loading import (
    build_species_list,
    extract_site,
    list_sites,
    load_site,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

N_COMPONENTS = 200
MIN_CLUSTER_SIZE = 15
MAX_PROTOTYPES = 12
LEIDEN_RESOLUTION = 1.0
MIN_COHESION = 0.3

_TIME_RE = re.compile(r"_(\d{6})\.ogg$")


def _extract_hour(filename: str) -> int | None:
    m = _TIME_RE.search(filename)
    return int(m.group(1)[:2]) if m else None


def _diel_period(hour: int) -> str:
    if 5 <= hour < 8:
        return "dawn"
    elif 8 <= hour < 17:
        return "day"
    elif 17 <= hour < 20:
        return "dusk"
    return "night"


class SitePrototypes(AnalysisTask):
    """Build site-state prototypes via Leiden clustering on soundscape windows."""

    def requires(self):
        return PartitionFeatures(**self._params())

    def output(self):
        return [
            self.artifact("site_prototypes.npz"),
            self.artifact("site_prototypes_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        perch_labels = self._load_perch_labels()
        sites = list_sites(self.features_dir)

        # =====================================================================
        # Step 1: per-site means + global mean + SVD whitening
        # =====================================================================
        logger.info("SitePrototypes: step 1 — per-site means + SVD")

        logit_dim = None
        site_means = {}
        site_counts = {}
        global_sum = None
        global_count = 0
        residual_samples = []

        for site in sites:
            df = load_site(site, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                global_sum = np.zeros(logit_dim, dtype=np.float64)

            site_mean = logits.mean(axis=0)
            site_means[site] = site_mean
            site_counts[site] = len(df)
            global_sum += logits.sum(axis=0)
            global_count += len(df)

            # Subsample residuals for SVD
            residuals = logits - site_mean[None, :]
            step = max(1, len(df) // 2500)  # ~2500 per site, ~50K total
            for j in range(0, len(df), step):
                residual_samples.append(residuals[j])

            logger.info("  %s: %d windows loaded", site, len(df))

        global_sc_mean = global_sum / global_count
        logger.info("SitePrototypes: %d sites, %d total windows", len(sites), global_count)

        res_matrix = np.array(residual_samples, dtype=np.float64)
        del residual_samples
        n_comp = min(N_COMPONENTS, res_matrix.shape[0] - 1, res_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(res_matrix)
        components = svd.components_
        whiten_scale = 1.0 / np.maximum(svd.singular_values_, 1e-8)
        explained = svd.explained_variance_ratio_.sum()
        logger.info("SitePrototypes: %d components, %.1f%% variance", n_comp, explained * 100)
        del res_matrix

        # =====================================================================
        # Step 2: per-site clustering
        # =====================================================================
        logger.info("SitePrototypes: step 2 — per-site clustering")

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError("Requires igraph and leidenalg.")

        all_prototypes = []
        proto_site_list = []
        proto_cluster_id = []
        per_site_info = {}

        for site in sites:
            n_windows = site_counts[site]

            if n_windows < MIN_CLUSTER_SIZE:
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                continue

            # Load this site's data
            df = load_site(site, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)
            hours = np.array([
                _extract_hour(str(fn)) if _extract_hour(str(fn)) is not None else -1
                for fn in df["filename"].to_list()
            ], dtype=np.int16)

            residuals = logits - site_means[site][None, :]
            projected = (residuals @ components.T) * whiten_scale[None, :]
            norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
            proj_normed = (projected / norms).astype(np.float32)

            # Dense Gram → Leiden
            gram = proj_normed @ proj_normed.T
            np.fill_diagonal(gram, 0.0)
            gram = np.maximum(gram, 0.0)
            sources, targets = np.triu_indices(n_windows, k=1)
            weights = gram[sources, targets]
            pos_mask = weights > 0
            del gram

            if pos_mask.sum() == 0:
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                continue

            g = ig.Graph(
                n=n_windows,
                edges=list(zip(sources[pos_mask].tolist(), targets[pos_mask].tolist())),
                directed=False,
            )
            g.es["weight"] = weights[pos_mask].tolist()
            del sources, targets, weights, pos_mask

            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=LEIDEN_RESOLUTION, seed=42,
            )
            labels = np.array(partition.membership)
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            # Community stats with cohesion + diel characterization
            communities = []
            for cl, count in zip(unique_labels, label_counts):
                cl_mask = labels == cl
                cl_proj = proj_normed[cl_mask]
                cl_gram = cl_proj @ cl_proj.T
                n_cl = cl_gram.shape[0]
                cohesion = (cl_gram.sum() - n_cl) / max(n_cl * (n_cl - 1), 1)

                cl_hours = [int(h) for h in hours[cl_mask] if h >= 0]
                diel_dist = {"dawn": 0, "day": 0, "dusk": 0, "night": 0}
                for h in cl_hours:
                    diel_dist[_diel_period(h)] += 1
                total_h = max(sum(diel_dist.values()), 1)
                diel_frac = {k: round(v / total_h, 3) for k, v in diel_dist.items()}

                if count < MIN_CLUSTER_SIZE:
                    status = "rejected_too_small"
                elif cohesion < MIN_COHESION:
                    status = "rejected_low_cohesion"
                else:
                    status = "pending"
                communities.append({
                    "id": int(cl), "size": int(count),
                    "cohesion": round(float(cohesion), 4), "status": status,
                    "diel_distribution": diel_frac,
                    "median_hour": int(np.median(cl_hours)) if cl_hours else None,
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

            if not accepted_ids:
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_cluster_passed_filters",
                    "community_stats": self._build_comm_stats(communities),
                    "communities": sorted(communities, key=lambda c: -c["size"]),
                }
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
            else:
                for cid in accepted_ids:
                    cluster_mean = residuals[labels == cid].mean(axis=0)
                    proto = (site_means[site] + cluster_mean - global_sc_mean).astype(np.float32)
                    all_prototypes.append(proto)
                    proto_site_list.append(site)
                    proto_cluster_id.append(cid)
                    for comm in communities:
                        if comm["id"] == cid and comm["status"] == "accepted":
                            comm["top_perch_taxa"] = self._top_taxa(proto, perch_labels)

                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": len(accepted_ids),
                    "fallback": None,
                    "community_stats": self._build_comm_stats(communities),
                    "communities": sorted(communities, key=lambda c: -c["size"]),
                }

            logger.info("  %s: %d windows → %d communities → %d prototypes",
                         site, n_windows, len(communities),
                         per_site_info[site]["n_prototypes"])

        # Stack
        prototype_matrix = np.stack(all_prototypes)
        proto_site_arr = np.array(proto_site_list)
        proto_cluster_id_arr = np.array(proto_cluster_id, dtype=np.int32)
        proto_norms = np.maximum(np.linalg.norm(prototype_matrix, axis=1, keepdims=True), 1e-8)
        prototype_matrix_normed = prototype_matrix / proto_norms
        total_protos = len(all_prototypes)

        logger.info("SitePrototypes: %d prototypes across %d sites (%.1f avg)",
                     total_protos, len(sites), total_protos / max(len(sites), 1))

        # =====================================================================
        # Step 3: species × site-state affinity
        # =====================================================================
        logger.info("SitePrototypes: computing species × site-state affinity")
        species_list = build_species_list(self.artifacts_dir)
        focal_data = np.load(Path(self.artifacts_dir) / "focal_profiles_raw.npz")
        sp_means_focal = focal_data["mean_profiles"].astype(np.float64)
        sp_count = focal_data["species_count"]
        has_sp = sp_count > 0
        global_focal_mean = sp_means_focal[has_sp].mean(axis=0)
        sp_centered = sp_means_focal.copy()
        sp_centered[has_sp] -= global_focal_mean[None, :]
        sp_norms = np.maximum(np.linalg.norm(sp_centered, axis=1, keepdims=True), 1e-8)
        affinity = (sp_centered / sp_norms) @ prototype_matrix_normed.T
        affinity[~has_sp] = 0

        # =====================================================================
        # Save
        # =====================================================================
        np.savez_compressed(
            self.output()[0].path,
            prototypes=prototype_matrix.astype(np.float32),
            prototypes_normed=prototype_matrix_normed.astype(np.float32),
            proto_site=proto_site_arr, proto_cluster_id=proto_cluster_id_arr,
            sites=np.array(sites),
            site_counts=np.array([site_counts[s] for s in sites], dtype=np.int64),
            global_soundscape_mean=global_sc_mean.astype(np.float32),
            whiten_components=components.astype(np.float32),
            whiten_scale=whiten_scale.astype(np.float32),
            explained_variance_ratio=svd.explained_variance_ratio_.astype(np.float32),
            species_site_affinity=affinity.astype(np.float32),
            species_list=np.array(species_list),
        )

        summary = {
            "n_sites": len(sites), "total_windows": global_count,
            "total_prototypes": total_protos,
            "avg_prototypes_per_site": round(total_protos / max(len(sites), 1), 2),
            "whiten_n_components": n_comp,
            "whiten_explained_variance": round(float(explained), 4),
            "min_cluster_size": MIN_CLUSTER_SIZE, "leiden_resolution": LEIDEN_RESOLUTION,
            "min_cohesion": MIN_COHESION,
            "per_site": per_site_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SitePrototypes: done")

    def _build_comm_stats(self, communities):
        sizes = [c["size"] for c in communities]
        cohesions = [c["cohesion"] for c in communities]
        return {
            "n_total": len(communities),
            "n_accepted": sum(1 for c in communities if c["status"] == "accepted"),
            "n_rejected_too_small": sum(1 for c in communities if c["status"] == "rejected_too_small"),
            "n_rejected_low_cohesion": sum(1 for c in communities if c["status"] == "rejected_low_cohesion"),
            "n_rejected_max_cap": sum(1 for c in communities if c["status"] == "rejected_max_cap"),
            "size_min": min(sizes), "size_max": max(sizes), "size_median": int(np.median(sizes)),
            "cohesion_min": round(min(cohesions), 4), "cohesion_max": round(max(cohesions), 4),
            "cohesion_median": round(float(np.median(cohesions)), 4),
        }

    def _top_taxa(self, prototype, perch_labels, n=10):
        abs_vals = np.abs(prototype)
        top_idx = np.argsort(-abs_vals)[:n]
        return [
            {"logit_index": int(i),
             "perch_species": perch_labels[int(i)] if int(i) < len(perch_labels) else f"logit_{i}",
             "value": round(float(prototype[i]), 4)}
            for i in top_idx
        ]

    def _load_perch_labels(self):
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
