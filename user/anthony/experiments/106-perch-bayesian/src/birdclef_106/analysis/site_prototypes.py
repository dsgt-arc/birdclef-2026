"""Site-state prototypes via Leiden clustering on within-site residuals.

Instead of a single mean profile per site, cluster soundscape windows within
each site to discover distinct acoustic states (dawn chorus, night insects,
rain, quiet midday, etc.). Each cluster centroid becomes a site-state prototype.

Memory optimization: stores projected vectors (200-d) not raw logits (14795-d).
Peak: ~127K × 200 × 4 = ~100 MB for projected vectors.
Prototype centroids computed via a final streaming pass over raw shards.

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
from birdclef_106.analysis.focal_profiles import FocalSpeciesProfiles
from birdclef_106.data_loading import (
    build_species_list,
    extract_site,
    load_unlabeled_shards,
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
        return FocalSpeciesProfiles(**self._params())

    def output(self):
        return [
            self.artifact("site_prototypes.npz"),
            self.artifact("site_prototypes_summary.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        perch_labels = self._load_perch_labels()

        # =====================================================================
        # Pass 1: per-site means + metadata (hours) — no raw logit storage
        # =====================================================================

        logger.info("SitePrototypes: pass 1 — per-site means")

        site_sums: dict[str, np.ndarray] = {}
        site_counts: dict[str, int] = {}
        site_hours: dict[str, list[int | None]] = {}
        logit_dim = None

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]

            filenames = shard["filename"].to_list()
            shard_sites = [extract_site(str(fn)) for fn in filenames]
            shard_hours = [_extract_hour(str(fn)) for fn in filenames]

            unique_sites = set(shard_sites)
            for site in unique_sites:
                if site not in site_sums:
                    site_sums[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_counts[site] = 0
                    site_hours[site] = []

            site_arr = np.array(shard_sites)
            for site in unique_sites:
                mask = site_arr == site
                idx = np.where(mask)[0]
                site_sums[site] += logits[idx].sum(axis=0)
                site_counts[site] += len(idx)
                for oi in idx:
                    site_hours[site].append(shard_hours[oi])

            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        sites = sorted(site_sums.keys())
        site_means = {s: site_sums[s] / site_counts[s] for s in sites}
        total_windows = sum(site_counts.values())
        global_sc_mean = sum(site_sums.values()) / total_windows

        logger.info("SitePrototypes: %d sites, %s windows",
                     len(sites), {s: site_counts[s] for s in sites})

        # =====================================================================
        # Pass 2: subsample within-site residuals for SVD
        # =====================================================================

        logger.info("SitePrototypes: pass 2 — subsampling residuals for SVD")

        svd_subsample_rate = max(1, total_windows // 50_000)
        residual_samples = []
        sample_counter = 0

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            filenames = shard["filename"].to_list()

            for j in range(len(shard)):
                site = extract_site(str(filenames[j]))
                if sample_counter % svd_subsample_rate == 0:
                    residual_samples.append(logits[j] - site_means[site])
                sample_counter += 1

            if i % 4 == 0:
                logger.info("  pass 2 shard %d done (%d samples)", i, len(residual_samples))

        residual_matrix = np.array(residual_samples, dtype=np.float64)
        del residual_samples

        n_comp = min(N_COMPONENTS, residual_matrix.shape[0] - 1, residual_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(residual_matrix)
        components = svd.components_
        whiten_scale = 1.0 / np.maximum(svd.singular_values_, 1e-8)
        explained = svd.explained_variance_ratio_.sum()
        logger.info("SitePrototypes: %d components, %.1f%% variance", n_comp, explained * 100)
        del residual_matrix

        # =====================================================================
        # Pass 3: project all windows to whitened 200-d space
        # Store projected vectors + site labels (compact: ~100 MB)
        # =====================================================================

        logger.info("SitePrototypes: pass 3 — projecting to whitened space")

        all_projected = []
        all_site_labels = []
        all_hours_flat = []

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            filenames = shard["filename"].to_list()

            shard_sites = [extract_site(str(fn)) for fn in filenames]
            shard_hours_list = [_extract_hour(str(fn)) for fn in filenames]

            # Vectorized: build residuals for whole shard
            site_mean_arr = np.array([site_means[s] for s in shard_sites])
            residuals = logits - site_mean_arr
            projected = (residuals @ components.T) * whiten_scale[None, :]

            all_projected.append(projected.astype(np.float32))
            all_site_labels.extend(shard_sites)
            all_hours_flat.extend(shard_hours_list)

            if i % 4 == 0:
                logger.info("  pass 3 shard %d done", i)

        all_projected = np.concatenate(all_projected)
        all_site_labels = np.array(all_site_labels)
        logger.info("SitePrototypes: projected %d windows to %d-d", len(all_projected), n_comp)

        # =====================================================================
        # Per-site Leiden clustering on projected vectors
        # =====================================================================

        logger.info("SitePrototypes: clustering")

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError("Requires igraph and leidenalg.")

        per_site_labels = {}  # site -> labels array
        per_site_accepted = {}  # site -> list of accepted cluster IDs or None
        per_site_info = {}

        for site in sites:
            mask = all_site_labels == site
            n_windows = int(mask.sum())

            if n_windows < MIN_CLUSTER_SIZE:
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                per_site_accepted[site] = None
                continue

            sp_projected = all_projected[mask]
            norms = np.maximum(np.linalg.norm(sp_projected, axis=1, keepdims=True), 1e-8)
            sp_normed = sp_projected / norms

            # Dense Gram matrix
            gram = sp_normed @ sp_normed.T
            np.fill_diagonal(gram, 0.0)
            gram = np.maximum(gram, 0.0)

            sources, targets = np.triu_indices(n_windows, k=1)
            weights = gram[sources, targets]
            pos_mask = weights > 0
            sources = sources[pos_mask]
            targets = targets[pos_mask]
            weights = weights[pos_mask]
            del gram

            if len(sources) == 0:
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                per_site_accepted[site] = None
                continue

            g = ig.Graph(n=n_windows,
                         edges=list(zip(sources.tolist(), targets.tolist())),
                         directed=False)
            g.es["weight"] = weights.tolist()
            del sources, targets, weights

            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=LEIDEN_RESOLUTION, seed=42,
            )
            labels = np.array(partition.membership)
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            # Community stats with cohesion + temporal characterization
            site_hour_arr = np.array([
                all_hours_flat[gi] for gi in np.where(mask)[0]
            ])
            all_communities = []
            for cl, count in zip(unique_labels, label_counts):
                cl_mask = labels == cl
                cl_proj = sp_normed[cl_mask]
                cl_gram = cl_proj @ cl_proj.T
                n_cl = cl_gram.shape[0]
                cohesion = (cl_gram.sum() - n_cl) / max(n_cl * (n_cl - 1), 1)

                cl_hours = [int(h) for h in site_hour_arr[cl_mask] if h is not None]
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

                all_communities.append({
                    "id": int(cl), "size": int(count),
                    "cohesion": round(float(cohesion), 4), "status": status,
                    "diel_distribution": diel_frac,
                    "median_hour": int(np.median(cl_hours)) if cl_hours else None,
                })

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

            if not accepted_ids:
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_cluster_passed_filters",
                    "community_stats": self._build_comm_stats(all_communities),
                    "communities": sorted(all_communities, key=lambda c: -c["size"]),
                }
                per_site_accepted[site] = None
            else:
                per_site_labels[site] = labels
                per_site_accepted[site] = accepted_ids
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": len(accepted_ids),
                    "fallback": None,
                    "community_stats": self._build_comm_stats(all_communities),
                    "communities": sorted(all_communities, key=lambda c: -c["size"]),
                }

            logger.info("  %s: %d windows → %d communities → %d prototypes",
                         site, n_windows, len(all_communities), max(n_proto, 1))

        del all_projected, all_site_labels, all_hours_flat

        # =====================================================================
        # Pass 4: compute prototype centroids from raw logits (streaming)
        # =====================================================================

        logger.info("SitePrototypes: pass 4 — computing prototype centroids")

        # Accumulators: (site, cluster_id) -> (sum_residual, count)
        cluster_sums = {}
        for site, accepted in per_site_accepted.items():
            if accepted is None:
                continue
            for cid in accepted:
                cluster_sums[(site, cid)] = [np.zeros(logit_dim, dtype=np.float64), 0]

        # Track per-site row counters to align with Leiden labels
        site_row_counters = {s: 0 for s in sites}

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            filenames = shard["filename"].to_list()

            for j in range(len(shard)):
                site = extract_site(str(filenames[j]))
                row_in_site = site_row_counters[site]
                site_row_counters[site] += 1

                if site not in per_site_labels:
                    continue
                labels = per_site_labels[site]
                cid = int(labels[row_in_site])
                key = (site, cid)
                if key in cluster_sums:
                    cluster_sums[key][0] += logits[j] - site_means[site]
                    cluster_sums[key][1] += 1

            if i % 4 == 0:
                logger.info("  pass 4 shard %d done", i)

        # Build prototypes
        all_prototypes = []
        proto_site_list = []
        proto_cluster_id = []
        proto_hours = []

        for site in sites:
            accepted = per_site_accepted.get(site)
            if accepted is None:
                prototype = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(prototype)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                hours = [h for h in site_hours[site] if h is not None]
                proto_hours.append(hours)
                continue

            for cid in accepted:
                s, c = cluster_sums[(site, cid)]
                cluster_mean = s / c if c > 0 else np.zeros(logit_dim)
                prototype = (site_means[site] + cluster_mean - global_sc_mean).astype(np.float32)
                all_prototypes.append(prototype)
                proto_site_list.append(site)
                proto_cluster_id.append(cid)

                # Get hours for this cluster from labels
                labels = per_site_labels[site]
                cl_mask = labels == cid
                site_hour_list = site_hours[site]
                cl_hours = [site_hour_list[k] for k in range(len(labels)) if cl_mask[k] and site_hour_list[k] is not None]
                proto_hours.append(cl_hours)

                # Add top taxa to accepted community in per_site_info
                for comm in per_site_info[site]["communities"]:
                    if comm["id"] == cid and comm["status"] == "accepted":
                        comm["top_perch_taxa"] = self._top_taxa(prototype, perch_labels, n=10)

        # Stack
        prototype_matrix = np.stack(all_prototypes)
        proto_site_arr = np.array(proto_site_list)
        proto_cluster_id_arr = np.array(proto_cluster_id, dtype=np.int32)

        proto_norms = np.maximum(
            np.linalg.norm(prototype_matrix, axis=1, keepdims=True), 1e-8
        )
        prototype_matrix_normed = prototype_matrix / proto_norms

        total_protos = len(all_prototypes)
        logger.info("SitePrototypes: %d total prototypes across %d sites (%.1f avg)",
                     total_protos, len(sites), total_protos / max(len(sites), 1))

        # =====================================================================
        # Species × Site-state affinity
        # =====================================================================

        logger.info("SitePrototypes: computing species × site-state affinity")

        species_list = build_species_list(self.artifacts_dir)
        n_species = len(species_list)

        focal_npz_path = Path(self.artifacts_dir) / "focal_profiles_raw.npz"
        focal_data = np.load(focal_npz_path)
        sp_means_focal = focal_data["mean_profiles"].astype(np.float64)
        sp_count = focal_data["species_count"]

        has_sp_data = sp_count > 0
        global_focal_mean = sp_means_focal[has_sp_data].mean(axis=0)
        sp_centered = sp_means_focal.copy()
        sp_centered[has_sp_data] -= global_focal_mean[None, :]
        sp_norms = np.maximum(np.linalg.norm(sp_centered, axis=1, keepdims=True), 1e-8)
        sp_normed = sp_centered / sp_norms

        affinity = sp_normed @ prototype_matrix_normed.T
        affinity[~has_sp_data] = 0

        logger.info("SitePrototypes: affinity shape %s, range [%.4f, %.4f]",
                     affinity.shape, affinity[has_sp_data].min(), affinity[has_sp_data].max())

        # =====================================================================
        # Save
        # =====================================================================

        np.savez_compressed(
            self.output()[0].path,
            prototypes=prototype_matrix.astype(np.float32),
            prototypes_normed=prototype_matrix_normed.astype(np.float32),
            proto_site=proto_site_arr,
            proto_cluster_id=proto_cluster_id_arr,
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
            "n_sites": len(sites),
            "total_windows": total_windows,
            "total_prototypes": total_protos,
            "avg_prototypes_per_site": round(total_protos / max(len(sites), 1), 2),
            "whiten_n_components": n_comp,
            "whiten_explained_variance": round(float(explained), 4),
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "leiden_resolution": LEIDEN_RESOLUTION,
            "per_site": per_site_info,
        }

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SitePrototypes: saved summary to %s", self.output()[1].path)

    def _build_comm_stats(self, communities):
        sizes = [c["size"] for c in communities]
        cohesions = [c["cohesion"] for c in communities]
        return {
            "n_total": len(communities),
            "n_accepted": sum(1 for c in communities if c["status"] == "accepted"),
            "n_rejected_too_small": sum(1 for c in communities if c["status"] == "rejected_too_small"),
            "n_rejected_low_cohesion": sum(1 for c in communities if c["status"] == "rejected_low_cohesion"),
            "n_rejected_max_cap": sum(1 for c in communities if c["status"] == "rejected_max_cap"),
            "size_min": min(sizes), "size_max": max(sizes),
            "size_median": int(np.median(sizes)),
            "cohesion_min": round(min(cohesions), 4),
            "cohesion_max": round(max(cohesions), 4),
            "cohesion_median": round(float(np.median(cohesions)), 4),
        }

    def _top_taxa(self, prototype: np.ndarray, perch_labels: list[str], n: int = 10) -> list[dict]:
        abs_vals = np.abs(prototype)
        top_idx = np.argsort(-abs_vals)[:n]
        return [
            {"logit_index": int(idx),
             "perch_species": perch_labels[int(idx)] if int(idx) < len(perch_labels) else f"logit_{idx}",
             "value": round(float(prototype[idx]), 4)}
            for idx in top_idx
        ]

    def _load_perch_labels(self) -> list[str]:
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
