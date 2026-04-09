"""Site-state prototypes via Leiden clustering on within-site residuals.

Pipeline:
    1. Pass 1: per-site means + metadata (hours) from unlabeled soundscapes.
    2. Pass 2: subsample residuals → SVD whitening transform.
    3. Pass 3: write per-site .npz files to scratch (projected + residuals + hours).
    4. Per site: load .npz, Leiden cluster, compute centroid prototypes.
    5. Compute species × site-state affinity matrix.

Per-site files on scratch (~7 GB total) make the pipeline robust to failure.

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

    def _cache_dir(self):
        return Path(self.features_dir).expanduser() / "site_proto_cache"

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        perch_labels = self._load_perch_labels()
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # Pass 1: per-site means
        # =====================================================================

        logger.info("SitePrototypes: pass 1 — per-site means")
        site_sums: dict[str, np.ndarray] = {}
        site_counts: dict[str, int] = {}
        logit_dim = None

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]
            filenames = shard["filename"].to_list()
            shard_sites = [extract_site(str(fn)) for fn in filenames]

            for site in set(shard_sites):
                if site not in site_sums:
                    site_sums[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_counts[site] = 0

            site_arr = np.array(shard_sites)
            for site in set(shard_sites):
                idx = np.where(site_arr == site)[0]
                site_sums[site] += logits[idx].sum(axis=0)
                site_counts[site] += len(idx)

            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        sites = sorted(site_sums.keys())
        site_means = {s: site_sums[s] / site_counts[s] for s in sites}
        total_windows = sum(site_counts.values())
        global_sc_mean = sum(site_sums.values()) / total_windows

        logger.info("SitePrototypes: %d sites, %s windows",
                     len(sites), {s: site_counts[s] for s in sites})

        # =====================================================================
        # Pass 2: subsample residuals → SVD
        # =====================================================================

        logger.info("SitePrototypes: pass 2 — SVD")
        svd_rate = max(1, total_windows // 50_000)
        residual_samples = []
        counter = 0

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            filenames = shard["filename"].to_list()
            for j in range(len(shard)):
                if counter % svd_rate == 0:
                    site = extract_site(str(filenames[j]))
                    residual_samples.append(logits[j] - site_means[site])
                counter += 1
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
        logger.info("SitePrototypes: %d components, %.1f%% variance", n_comp, explained * 100)
        del res_matrix

        # =====================================================================
        # Pass 3: write per-site .npz to scratch
        # Each file: {projected, residuals, hours} — total ~7 GB
        # =====================================================================

        logger.info("SitePrototypes: pass 3 — writing per-site caches")

        site_projected: dict[str, list] = {s: [] for s in sites}
        site_residuals: dict[str, list] = {s: [] for s in sites}
        site_hours: dict[str, list] = {s: [] for s in sites}

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            filenames = shard["filename"].to_list()
            shard_sites = [extract_site(str(fn)) for fn in filenames]
            shard_hours = [_extract_hour(str(fn)) for fn in filenames]

            # Vectorized residuals + projection per site
            site_arr = np.array(shard_sites)
            for site in set(shard_sites):
                idx = np.where(site_arr == site)[0]
                res = logits[idx] - site_means[site]
                proj = ((res @ components.T) * whiten_scale[None, :]).astype(np.float32)
                res_f32 = res.astype(np.float32)
                site_projected[site].append(proj)
                site_residuals[site].append(res_f32)
                for oi in idx:
                    site_hours[site].append(shard_hours[oi] if shard_hours[oi] is not None else -1)

            if i % 4 == 0:
                logger.info("  pass 3 shard %d done", i)

        for site in sites:
            if not site_projected[site]:
                continue
            np.savez(
                cache_dir / f"{site}.npz",
                projected=np.concatenate(site_projected[site]),
                residuals=np.concatenate(site_residuals[site]),
                hours=np.array(site_hours[site], dtype=np.int16),
            )
        del site_projected, site_residuals, site_hours
        logger.info("SitePrototypes: wrote %d site caches to %s", len(sites), cache_dir)

        # =====================================================================
        # Step 4: per-site clustering from cached files
        # =====================================================================

        logger.info("SitePrototypes: step 4 — clustering")

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
            cache_path = cache_dir / f"{site}.npz"
            if not cache_path.exists():
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                per_site_info[site] = {
                    "n_windows": site_counts[site], "n_prototypes": 1,
                    "fallback": "no_cache",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                continue

            data = np.load(cache_path)
            projected = data["projected"]
            residuals = data["residuals"]
            hours = data["hours"]
            n_windows = projected.shape[0]

            if n_windows < MIN_CLUSTER_SIZE:
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
                continue

            norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
            proj_normed = projected / norms

            # Dense Gram → Leiden
            gram = proj_normed @ proj_normed.T
            np.fill_diagonal(gram, 0.0)
            gram = np.maximum(gram, 0.0)

            sources, targets = np.triu_indices(n_windows, k=1)
            weights = gram[sources, targets]
            pos_mask = weights > 0
            del gram

            if pos_mask.sum() == 0:
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": 1,
                    "fallback": "no_positive_edges",
                    "community_stats": {"n_total": 0}, "communities": [],
                }
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

            # Accept in size order
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
                proto = (site_means[site] - global_sc_mean).astype(np.float32)
                all_prototypes.append(proto)
                proto_site_list.append(site)
                proto_cluster_id.append(0)
                fallback = "no_cluster_passed_filters"
            else:
                for cid in accepted_ids:
                    cl_res = residuals[labels == cid].astype(np.float64)
                    cluster_mean = cl_res.mean(axis=0)
                    proto = (site_means[site] + cluster_mean - global_sc_mean).astype(np.float32)
                    all_prototypes.append(proto)
                    proto_site_list.append(site)
                    proto_cluster_id.append(cid)

                    # Add top taxa to community info
                    for comm in communities:
                        if comm["id"] == cid and comm["status"] == "accepted":
                            comm["top_perch_taxa"] = self._top_taxa(proto, perch_labels)
                fallback = None

            sizes = [c["size"] for c in communities]
            cohesions = [c["cohesion"] for c in communities]
            per_site_info[site] = {
                "n_windows": n_windows,
                "n_prototypes": max(len(accepted_ids), 1),
                "fallback": fallback,
                "community_stats": {
                    "n_total": len(communities),
                    "n_accepted": len(accepted_ids),
                    "n_rejected_too_small": sum(1 for c in communities if c["status"] == "rejected_too_small"),
                    "n_rejected_low_cohesion": sum(1 for c in communities if c["status"] == "rejected_low_cohesion"),
                    "n_rejected_max_cap": sum(1 for c in communities if c["status"] == "rejected_max_cap"),
                    "size_min": min(sizes), "size_max": max(sizes),
                    "size_median": int(np.median(sizes)),
                    "cohesion_min": round(min(cohesions), 4),
                    "cohesion_max": round(max(cohesions), 4),
                    "cohesion_median": round(float(np.median(cohesions)), 4),
                },
                "communities": sorted(communities, key=lambda c: -c["size"]),
            }

            logger.info("  %s: %d windows → %d communities → %d prototypes",
                         site, n_windows, len(communities), max(len(accepted_ids), 1))

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
        # Species × Site-state affinity
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

        logger.info("SitePrototypes: affinity shape %s", affinity.shape)

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
            "min_cohesion": MIN_COHESION,
            "cache_dir": str(cache_dir),
            "per_site": per_site_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SitePrototypes: done")

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
