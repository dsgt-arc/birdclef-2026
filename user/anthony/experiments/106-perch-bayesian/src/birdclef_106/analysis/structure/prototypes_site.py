"""Site-state prototypes: NMF on logits, k-means on embeddings.

For each site, decompose soundscape windows into K prototype patterns
representing distinct acoustic states (dawn chorus, night insects, etc.).

Fixed K per site gives a clean (L, K, P) tensor.

Depends on PartitionFeatures for per-site parquets.

Outputs:
    site_prototypes.npz — prototype tensors, affinity matrix, metadata
    site_prototypes_summary.json — per-site cluster info, diel characterization
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.partition import PartitionFeatures
from birdclef_106.data_loading import (
    build_species_list,
    list_sites,
    load_site,
    logits_to_numpy,
    embeddings_to_numpy,
)

logger = logging.getLogger(__name__)

N_PROTOTYPES = 6
MIN_WINDOWS = 20

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
    """Build site-state prototypes via NMF (logits) + k-means (embeddings)."""

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
        n_sites = len(sites)
        K = N_PROTOTYPES

        logger.info("SitePrototypes: decomposing %d sites (K=%d)", n_sites, K)

        logit_dim = None
        emb_dim = None
        global_logit_sum = None
        global_emb_sum = None
        global_count = 0

        site_logit_protos = {}
        site_emb_protos = {}
        site_counts = {}
        per_site_info = {}

        for site in sites:
            df = load_site(site, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)
            emb = embeddings_to_numpy(df).astype(np.float64)
            n_windows = len(df)
            site_counts[site] = n_windows

            if logit_dim is None:
                logit_dim = logits.shape[1]
                emb_dim = emb.shape[1]
                global_logit_sum = np.zeros(logit_dim, dtype=np.float64)
                global_emb_sum = np.zeros(emb_dim, dtype=np.float64)

            global_logit_sum += logits.sum(axis=0)
            global_emb_sum += emb.sum(axis=0)
            global_count += n_windows

            # Extract hours for diel characterization
            filenames = df["filename"].to_list()
            hours = [_extract_hour(str(fn)) for fn in filenames]

            if n_windows < MIN_WINDOWS:
                site_logit_protos[site] = np.tile(logits.mean(axis=0), (K, 1))
                site_emb_protos[site] = np.tile(emb.mean(axis=0), (K, 1))
                per_site_info[site] = {
                    "n_windows": n_windows, "n_prototypes": K,
                    "method": "mean_fallback", "clusters": [],
                }
                continue

            k_actual = min(K, n_windows - 1)

            # NMF on logits
            logits_nn = np.maximum(logits, 0).astype(np.float64)
            try:
                nmf = NMF(n_components=k_actual, init="nndsvda", max_iter=300,
                          random_state=42)
                W_nmf = nmf.fit_transform(logits_nn)
                H = nmf.components_
            except Exception as e:
                logger.warning("  %s: NMF failed (%s), using mean", site, e)
                H = np.tile(logits.mean(axis=0), (k_actual, 1))
                W_nmf = np.ones((n_windows, k_actual)) / k_actual

            if H.shape[0] < K:
                H = np.vstack([H, np.tile(H.mean(axis=0), (K - H.shape[0], 1))])
            site_logit_protos[site] = H

            # K-means on embeddings
            try:
                km = KMeans(n_clusters=k_actual, random_state=42, n_init=3)
                labels = km.fit_predict(emb)
                centers = km.cluster_centers_
            except Exception as e:
                logger.warning("  %s: k-means failed (%s), using mean", site, e)
                centers = np.tile(emb.mean(axis=0), (k_actual, 1))
                labels = np.zeros(n_windows, dtype=int)

            if centers.shape[0] < K:
                centers = np.vstack([centers, np.tile(centers.mean(axis=0), (K - centers.shape[0], 1))])
            site_emb_protos[site] = centers

            # Diel characterization per cluster (from k-means labels)
            clusters_info = []
            for ci in range(k_actual):
                cl_mask = labels == ci
                cl_hours = [hours[j] for j in range(n_windows) if cl_mask[j] and hours[j] is not None]
                diel_dist = {"dawn": 0, "day": 0, "dusk": 0, "night": 0}
                for h in cl_hours:
                    diel_dist[_diel_period(h)] += 1
                total_h = max(sum(diel_dist.values()), 1)
                diel_frac = {k: round(v / total_h, 3) for k, v in diel_dist.items()}

                # Top Perch taxa for this cluster's NMF component
                top_taxa = []
                if ci < W_nmf.shape[1]:
                    component = H[ci] if ci < H.shape[0] else H[0]
                    top_idx = np.argsort(-np.abs(component))[:10]
                    top_taxa = [
                        {"logit_index": int(idx),
                         "perch_species": perch_labels[int(idx)] if int(idx) < len(perch_labels) else f"logit_{idx}",
                         "value": round(float(component[idx]), 4)}
                        for idx in top_idx
                    ]

                clusters_info.append({
                    "id": ci, "size": int(cl_mask.sum()),
                    "diel_distribution": diel_frac,
                    "median_hour": int(np.median(cl_hours)) if cl_hours else None,
                    "top_perch_taxa": top_taxa,
                })

            per_site_info[site] = {
                "n_windows": n_windows, "n_prototypes": K,
                "method": "nmf+kmeans", "clusters": clusters_info,
            }

            logger.info("  %s: %d windows → %d prototypes", site, n_windows, K)

        # Global means
        global_logit_mean = global_logit_sum / global_count
        global_emb_mean = global_emb_sum / global_count

        # =====================================================================
        # Build prototype tensors (L, K, P) and (L, K, E)
        # =====================================================================

        proto_logits = np.zeros((n_sites, K, logit_dim), dtype=np.float32)
        proto_emb = np.zeros((n_sites, K, emb_dim), dtype=np.float32)

        for li, site in enumerate(sites):
            if site in site_logit_protos:
                proto_logits[li] = site_logit_protos[site].astype(np.float32)[:K]
                proto_emb[li] = site_emb_protos[site].astype(np.float32)[:K]

        # Center and normalize
        proto_logits_centered = proto_logits - global_logit_mean[None, None, :]
        norms = np.maximum(np.linalg.norm(proto_logits_centered, axis=2, keepdims=True), 1e-8)
        proto_logits_normed = proto_logits_centered / norms

        proto_emb_centered = proto_emb - global_emb_mean[None, None, :]
        norms = np.maximum(np.linalg.norm(proto_emb_centered, axis=2, keepdims=True), 1e-8)
        proto_emb_normed = proto_emb_centered / norms

        # =====================================================================
        # Species × Site-state affinity
        # =====================================================================
        logger.info("SitePrototypes: computing species × site-state affinity")

        species_list = build_species_list(self.artifacts_dir)
        n_species = len(species_list)

        focal_data = np.load(Path(self.artifacts_dir) / "focal_profiles_raw.npz")
        sp_means = focal_data["mean_profiles"].astype(np.float64)
        sp_count = focal_data["species_count"]
        has_sp = sp_count > 0
        global_focal_mean = sp_means[has_sp].mean(axis=0)
        sp_centered = sp_means.copy()
        sp_centered[has_sp] -= global_focal_mean[None, :]
        sp_normed = normalize(sp_centered, norm="l2")

        # Flatten site protos to (L*K, P) for matmul
        flat_protos = proto_logits_normed.reshape(-1, logit_dim)
        affinity = sp_normed @ flat_protos.T  # (S, L*K)
        affinity[~has_sp] = 0

        # Also compute (S, L) by max over K per site
        affinity_3d = affinity.reshape(n_species, n_sites, K)
        affinity_max = affinity_3d.max(axis=2)  # (S, L)

        logger.info("SitePrototypes: affinity shape (%d species, %d sites, K=%d)",
                     n_species, n_sites, K)

        # =====================================================================
        # Save
        # =====================================================================
        np.savez_compressed(
            self.output()[0].path,
            proto_logits=proto_logits,
            proto_logits_normed=proto_logits_normed.astype(np.float32),
            proto_emb=proto_emb,
            proto_emb_normed=proto_emb_normed.astype(np.float32),
            # Flat versions for PrototypeRidge compatibility
            prototypes_normed=flat_protos.astype(np.float32),
            proto_site=np.repeat(np.array(sites), K),
            proto_cluster_id=np.tile(np.arange(K, dtype=np.int32), n_sites),
            # Affinity
            species_site_affinity=affinity.astype(np.float32),
            species_site_affinity_max=affinity_max.astype(np.float32),
            # Metadata
            sites=np.array(sites),
            site_counts=np.array([site_counts.get(s, 0) for s in sites], dtype=np.int64),
            global_soundscape_mean=global_logit_mean.astype(np.float32),
            global_emb_mean=global_emb_mean.astype(np.float32),
            species_list=np.array(species_list),
            n_prototypes=K,
        )

        summary = {
            "n_sites": n_sites,
            "total_windows": global_count,
            "n_prototypes_per_site": K,
            "total_prototypes": n_sites * K,
            "decomposition": {"logits": "NMF", "embeddings": "k-means"},
            "per_site": per_site_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("SitePrototypes: done")

    def _load_perch_labels(self):
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
