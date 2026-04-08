"""Site-state prototypes via Leiden clustering on within-site residuals.

Instead of a single mean profile per site, cluster soundscape windows within
each site to discover distinct acoustic states (dawn chorus, night insects,
rain, quiet midday, etc.). Each cluster centroid becomes a site-state prototype.

This replaces the fixed diel period binning (dawn/day/dusk/night) with
data-driven temporal/ecological states.

Pipeline:
    1. Pass 1: per-site means from unlabeled soundscapes.
    2. Pass 2: collect within-site residuals, subsample for SVD.
    3. SVD on pooled within-site residuals → whitening transform.
    4. Per site: whiten residuals, k-NN cosine graph, Leiden → prototypes.
    5. Characterize each prototype: dominant Perch taxa, temporal distribution.
    6. Compose species × site-state affinity matrix.

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

# Whitening PCA components
N_COMPONENTS = 200

# Minimum cluster size to become a prototype
MIN_CLUSTER_SIZE = 15

# Maximum prototypes per site
MAX_PROTOTYPES = 12

# Leiden resolution
LEIDEN_RESOLUTION = 1.0

# Minimum mean pairwise cosine within a cluster to keep it
MIN_COHESION = 0.3

# Hour boundaries for diel characterization (not used for clustering)
DIEL_BINS = {"night": (0, 5), "dawn": (5, 8), "day": (8, 17), "dusk": (17, 20), "night2": (20, 24)}

# Filename time pattern: BC2026_Train_{id}_{site}_{date}_{HHMMSS}.ogg
_TIME_RE = re.compile(r"_(\d{6})\.ogg$")


def _extract_hour(filename: str) -> int | None:
    """Extract hour from soundscape filename (HHMMSS suffix)."""
    m = _TIME_RE.search(filename)
    if m:
        return int(m.group(1)[:2])
    return None


def _diel_period(hour: int) -> str:
    """Map hour to diel period for characterization."""
    if 5 <= hour < 8:
        return "dawn"
    elif 8 <= hour < 17:
        return "day"
    elif 17 <= hour < 20:
        return "dusk"
    else:
        return "night"


class SitePrototypes(AnalysisTask):
    """Build site-state prototypes via Leiden clustering on soundscape windows."""

    def requires(self):
        # Needs focal profiles for species × site-state composition
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
        # Pass 1: per-site means and window metadata
        # =====================================================================

        logger.info("SitePrototypes: pass 1 — per-site means")

        site_windows: dict[str, list[dict]] = {}  # site -> [{logits, hour, filename}]
        site_sums: dict[str, np.ndarray] = {}
        site_counts: dict[str, int] = {}
        logit_dim = None

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard).astype(np.float64)
            if logit_dim is None:
                logit_dim = logits.shape[1]

            filenames = shard["filename"].to_list()
            # Vectorized metadata extraction
            shard_sites = [extract_site(str(fn)) for fn in filenames]
            shard_hours = [_extract_hour(str(fn)) for fn in filenames]

            # Group by site for vectorized sum
            unique_sites = set(shard_sites)
            for site in unique_sites:
                if site not in site_sums:
                    site_sums[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_counts[site] = 0
                    site_windows[site] = []

            site_arr = np.array(shard_sites)
            for site in unique_sites:
                mask = site_arr == site
                idx = np.where(mask)[0]
                site_sums[site] += logits[idx].sum(axis=0)
                site_counts[site] += len(idx)
                # Store per-window data for clustering
                logits_f32 = logits[idx].astype(np.float32)
                for j, oi in enumerate(idx):
                    site_windows[site].append({
                        "logits": logits_f32[j],
                        "hour": shard_hours[oi],
                        "filename": str(filenames[oi]),
                    })

            if i % 4 == 0:
                logger.info("  pass 1 shard %d done", i)

        sites = sorted(site_windows.keys())
        site_means = {}
        for site in sites:
            site_means[site] = site_sums[site] / site_counts[site]

        logger.info(
            "SitePrototypes: %d sites, %s windows",
            len(sites),
            {s: site_counts[s] for s in sites},
        )

        # =====================================================================
        # Compute within-site residuals, subsample for SVD
        # =====================================================================

        logger.info("SitePrototypes: computing within-site residuals for SVD")

        max_svd_samples = 50_000
        total_windows = sum(site_counts.values())
        svd_subsample_rate = max(1, total_windows // max_svd_samples)

        residual_samples = []
        row_counter = 0

        # Also compute residuals per site for clustering (stored in-place)
        for site in sites:
            mean = site_means[site]
            for w in site_windows[site]:
                residual = w["logits"].astype(np.float64) - mean
                w["residual"] = residual.astype(np.float32)

                if row_counter % svd_subsample_rate == 0:
                    residual_samples.append(residual)
                row_counter += 1

        residual_matrix = np.array(residual_samples, dtype=np.float64)
        logger.info(
            "SitePrototypes: SVD on %d residual samples (%d dims)",
            residual_matrix.shape[0], residual_matrix.shape[1],
        )

        # =====================================================================
        # Fit whitening transform
        # =====================================================================

        n_comp = min(N_COMPONENTS, residual_matrix.shape[0] - 1, residual_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(residual_matrix)

        singular_values = svd.singular_values_
        components = svd.components_  # (n_comp, logit_dim)
        whiten_scale = 1.0 / np.maximum(singular_values, 1e-8)

        explained = svd.explained_variance_ratio_.sum()
        logger.info(
            "SitePrototypes: %d components explain %.1f%% of within-site variance",
            n_comp, explained * 100,
        )

        del residual_matrix, residual_samples

        # =====================================================================
        # Per-site: whiten, k-NN graph, Leiden clustering
        # =====================================================================

        logger.info("SitePrototypes: clustering %d sites", len(sites))

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "Site prototypes require igraph and leidenalg. "
                "Install with: uv pip install igraph leidenalg"
            )

        # Global soundscape mean for centering in compositions
        global_sc_mean = sum(site_sums.values()) / total_windows

        all_prototypes = []  # (logit_dim,) arrays
        proto_site = []  # site ID per prototype
        proto_cluster_id = []  # cluster ID within site
        proto_hours = []  # hour distribution per prototype
        per_site_info = {}

        for site in sites:
            windows = site_windows[site]
            n_windows = len(windows)

            if n_windows < MIN_CLUSTER_SIZE:
                # Fall back to single mean
                prototype = site_means[site] - global_sc_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_site.append(site)
                proto_cluster_id.append(0)
                hours = [w["hour"] for w in windows if w["hour"] is not None]
                proto_hours.append(hours)
                per_site_info[site] = {
                    "n_windows": n_windows,
                    "n_kept": n_windows,
                    "n_prototypes": 1,
                    "fallback": "too_few_windows",
                    "clusters": [{"id": 0, "size": n_windows}],
                }
                continue

            # Stack residuals and whiten
            residuals = np.stack([w["residual"] for w in windows]).astype(np.float64)
            projected = (residuals @ components.T) * whiten_scale[None, :]

            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            projected_normed = projected / norms

            # Dense Mahalanobis-weighted similarity graph
            gram = projected_normed @ projected_normed.T  # (N_site, N_site)
            np.fill_diagonal(gram, 0.0)
            gram = np.maximum(gram, 0.0)

            # Build igraph from dense weighted adjacency
            sources, targets = np.triu_indices(n_windows, k=1)
            weights = gram[sources, targets]
            pos_mask = weights > 0
            sources = sources[pos_mask]
            targets = targets[pos_mask]
            weights = weights[pos_mask]

            if len(sources) == 0:
                prototype = site_means[site] - global_sc_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_site.append(site)
                proto_cluster_id.append(0)
                hours = [w["hour"] for w in windows if w["hour"] is not None]
                proto_hours.append(hours)
                per_site_info[site] = {
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

                gram = cl_projected @ cl_projected.T
                n_cl = gram.shape[0]
                cohesion = (gram.sum() - n_cl) / max(n_cl * (n_cl - 1), 1)

                # Temporal characterization (for all communities, not just accepted)
                cl_indices = np.where(cl_mask)[0]
                cluster_hours = [
                    windows[oi]["hour"] for oi in cl_indices
                    if windows[oi]["hour"] is not None
                ]
                diel_dist = {"dawn": 0, "day": 0, "dusk": 0, "night": 0}
                for h in cluster_hours:
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
                    "id": int(cl),
                    "size": int(count),
                    "cohesion": round(float(cohesion), 4),
                    "status": status,
                    "diel_distribution": diel_frac,
                    "median_hour": int(np.median(cluster_hours)) if cluster_hours else None,
                })

            # Accept communities in size order, up to MAX_PROTOTYPES
            n_proto = 0
            for comm in sorted(all_communities, key=lambda c: -c["size"]):
                if comm["status"] != "pending":
                    continue
                if n_proto >= MAX_PROTOTYPES:
                    comm["status"] = "rejected_max_cap"
                    continue
                comm["status"] = "accepted"
                n_proto += 1

            # Build prototypes from accepted communities
            for comm in sorted(all_communities, key=lambda c: -c["size"]):
                if comm["status"] != "accepted":
                    continue
                cl = comm["id"]
                cl_mask = labels == cl
                cl_indices = np.where(cl_mask)[0]

                cluster_residual_mean = np.stack(
                    [windows[oi]["residual"] for oi in cl_indices]
                ).astype(np.float64).mean(axis=0)
                prototype = (site_means[site] + cluster_residual_mean) - global_sc_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_site.append(site)
                proto_cluster_id.append(comm["id"])

                cluster_hours = [
                    windows[oi]["hour"] for oi in cl_indices
                    if windows[oi]["hour"] is not None
                ]
                proto_hours.append(cluster_hours)

                # Add top taxa to accepted community info
                top_taxa = self._top_taxa(prototype, perch_labels, n=10)
                comm["top_perch_taxa"] = top_taxa

            if n_proto == 0:
                prototype = site_means[site] - global_sc_mean
                all_prototypes.append(prototype.astype(np.float32))
                proto_site.append(site)
                proto_cluster_id.append(0)
                hours = [w["hour"] for w in windows if w["hour"] is not None]
                proto_hours.append(hours)
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

            per_site_info[site] = {
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

            logger.info(
                "  %s: %d windows → %d communities → %d prototypes",
                site, n_windows, len(all_communities), n_proto,
            )

        # Free per-window data
        del site_windows

        # Stack prototypes
        prototype_matrix = np.stack(all_prototypes)  # (total_protos, logit_dim)
        proto_site_arr = np.array(proto_site)
        proto_cluster_id_arr = np.array(proto_cluster_id, dtype=np.int32)

        # Normalize for cosine scoring
        proto_norms = np.maximum(
            np.linalg.norm(prototype_matrix, axis=1, keepdims=True), 1e-8
        )
        prototype_matrix_normed = prototype_matrix / proto_norms

        total_protos = len(all_prototypes)
        logger.info(
            "SitePrototypes: %d total prototypes across %d sites (%.1f avg)",
            total_protos, len(sites), total_protos / max(len(sites), 1),
        )

        # =====================================================================
        # Species × Site-state affinity: (S×K_s, P) @ (L×K_l, P)^T
        # Use single species means for now; multi-prototype version composes later
        # =====================================================================

        logger.info("SitePrototypes: computing species × site-state affinity")

        species_list = build_species_list(self.artifacts_dir)
        n_species = len(species_list)

        # Load focal species profiles
        focal_npz_path = Path(self.artifacts_dir) / "focal_profiles_raw.npz"
        focal_data = np.load(focal_npz_path)
        sp_means = focal_data["mean_profiles"].astype(np.float64)  # (n_species, P)
        sp_count = focal_data["species_count"]

        # Center species profiles by global focal mean
        has_sp_data = sp_count > 0
        global_focal_mean = sp_means[has_sp_data].mean(axis=0)
        sp_centered = sp_means.copy()
        sp_centered[has_sp_data] -= global_focal_mean[None, :]
        sp_norms = np.maximum(np.linalg.norm(sp_centered, axis=1, keepdims=True), 1e-8)
        sp_normed = sp_centered / sp_norms

        # Affinity: (n_species, total_protos) — species vs site-state cosine
        affinity = sp_normed @ prototype_matrix_normed.T

        # Zero out species without data
        affinity[~has_sp_data] = 0

        logger.info(
            "SitePrototypes: affinity matrix shape %s, range [%.4f, %.4f]",
            affinity.shape, affinity[has_sp_data].min(), affinity[has_sp_data].max(),
        )

        # =====================================================================
        # Save outputs
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
        logger.info("SitePrototypes: saved arrays to %s", self.output()[0].path)

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

    def _top_taxa(self, prototype: np.ndarray, perch_labels: list[str], n: int = 10) -> list[dict]:
        """Top Perch taxa by absolute activation in a prototype vector."""
        abs_vals = np.abs(prototype)
        top_idx = np.argsort(-abs_vals)[:n]
        result = []
        for idx in top_idx:
            name = perch_labels[int(idx)] if int(idx) < len(perch_labels) else f"logit_{idx}"
            result.append({
                "logit_index": int(idx),
                "perch_species": name,
                "value": round(float(prototype[idx]), 4),
            })
        return result

    def _load_perch_labels(self) -> list[str]:
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
