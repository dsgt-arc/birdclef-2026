"""Site-state prototypes: NMF on logits, k-means on embeddings.

Each site is its own Luigi task → parallel, idempotent.
SitePrototypes aggregates, characterizes diel, computes species affinity.

Output per site: artifacts/prototypes/site/{site}.npz
Aggregated: artifacts/site_prototypes.npz + site_prototypes_summary.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import luigi
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


class SingleSitePrototype(AnalysisTask):
    """NMF + k-means decomposition for one site."""

    site = luigi.Parameter()

    def requires(self):
        return PartitionFeatures(**self._params())

    def output(self):
        return self.artifact(f"prototypes/site/{self.site}.npz")

    def run(self):
        self.ensure_parent(self.output())
        K = N_PROTOTYPES

        df = load_site(self.site, self.features_dir)
        logits = logits_to_numpy(df).astype(np.float64)
        emb = embeddings_to_numpy(df).astype(np.float64)
        n_windows = len(df)
        filenames = df["filename"].to_list()
        hours = [_extract_hour(str(fn)) for fn in filenames]

        info = {"n_windows": n_windows, "n_prototypes": K}

        if n_windows < MIN_WINDOWS:
            H = np.tile(logits.mean(axis=0), (K, 1))
            centers = np.tile(emb.mean(axis=0), (K, 1))
            labels = np.zeros(n_windows, dtype=int)
            info["method"] = "mean_fallback"
        else:
            k_actual = min(K, n_windows - 1)

            # NMF on logits
            logits_nn = np.maximum(logits, 0).astype(np.float64)
            try:
                nmf = NMF(n_components=k_actual, init="nndsvda", max_iter=300,
                          random_state=42)
                nmf.fit_transform(logits_nn)
                H = nmf.components_
                info["nmf_error"] = round(float(nmf.reconstruction_err_), 4)
            except Exception as e:
                logger.warning("%s: NMF failed (%s), using mean", self.site, e)
                H = np.tile(logits.mean(axis=0), (k_actual, 1))

            if H.shape[0] < K:
                H = np.vstack([H, np.tile(H.mean(axis=0), (K - H.shape[0], 1))])

            # K-means on embeddings (also gives cluster labels for diel)
            try:
                km = KMeans(n_clusters=k_actual, random_state=42, n_init=3)
                labels = km.fit_predict(emb)
                centers = km.cluster_centers_
                info["kmeans_inertia"] = round(float(km.inertia_), 4)
            except Exception as e:
                logger.warning("%s: k-means failed (%s), using mean", self.site, e)
                centers = np.tile(emb.mean(axis=0), (k_actual, 1))
                labels = np.zeros(n_windows, dtype=int)

            if centers.shape[0] < K:
                centers = np.vstack([centers, np.tile(centers.mean(axis=0), (K - centers.shape[0], 1))])

            info["method"] = "nmf+kmeans"

        # Diel characterization per k-means cluster
        clusters_info = []
        for ci in range(min(K, max(labels) + 1)):
            cl_mask = labels == ci
            cl_hours = [hours[j] for j in range(n_windows) if cl_mask[j] and hours[j] is not None]
            diel_dist = {"dawn": 0, "day": 0, "dusk": 0, "night": 0}
            for h in cl_hours:
                diel_dist[_diel_period(h)] += 1
            total_h = max(sum(diel_dist.values()), 1)
            clusters_info.append({
                "id": ci, "size": int(cl_mask.sum()),
                "diel_distribution": {k: round(v / total_h, 3) for k, v in diel_dist.items()},
                "median_hour": int(np.median(cl_hours)) if cl_hours else None,
            })
        info["clusters"] = clusters_info

        np.savez(
            self.output().path,
            proto_logits=H.astype(np.float32),
            proto_emb=centers.astype(np.float32),
            logit_sum=logits.sum(axis=0),
            emb_sum=emb.sum(axis=0),
            n_windows=n_windows,
            info=json.dumps(info),
        )


class SitePrototypes(AnalysisTask):
    """Aggregate per-site prototypes, compute species × site-state affinity."""

    def requires(self):
        from birdclef_106.analysis.structure.profiles_focal import FocalSpeciesProfiles
        sites = list_sites(self.features_dir)
        p = self._params()
        return {
            "focal_profiles": FocalSpeciesProfiles(**p),
            **{site: SingleSitePrototype(site=site, **p) for site in sites},
        }

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

        logit_dim = emb_dim = None
        global_logit_sum = global_emb_sum = None
        global_count = 0
        per_site_info = {}
        proto_logits = proto_emb = None

        for li, site in enumerate(sites):
            path = Path(self.artifacts_dir) / "prototypes" / "site" / f"{site}.npz"
            if not path.exists():
                per_site_info[site] = {"n_windows": 0, "n_prototypes": 0, "method": "none"}
                continue

            data = np.load(path, allow_pickle=True)
            H = data["proto_logits"]
            centers = data["proto_emb"]
            n_windows = int(data["n_windows"])
            info = json.loads(str(data["info"]))

            if logit_dim is None:
                logit_dim = H.shape[1]
                emb_dim = centers.shape[1]
                global_logit_sum = np.zeros(logit_dim, dtype=np.float64)
                global_emb_sum = np.zeros(emb_dim, dtype=np.float64)
                proto_logits = np.zeros((n_sites, K, logit_dim), dtype=np.float32)
                proto_emb = np.zeros((n_sites, K, emb_dim), dtype=np.float32)

            global_logit_sum += data["logit_sum"].astype(np.float64)
            global_emb_sum += data["emb_sum"].astype(np.float64)
            global_count += n_windows
            proto_logits[li] = H[:K]
            proto_emb[li] = centers[:K]

            # Add top NMF taxa per cluster
            for ci, cl in enumerate(info.get("clusters", [])):
                if ci < H.shape[0]:
                    top_idx = np.argsort(-np.abs(H[ci]))[:10]
                    cl["top_perch_taxa"] = [
                        {"logit_index": int(idx),
                         "perch_species": perch_labels[int(idx)] if int(idx) < len(perch_labels) else f"logit_{idx}",
                         "value": round(float(H[ci][idx]), 4)}
                        for idx in top_idx
                    ]

            per_site_info[site] = info

        global_logit_mean = global_logit_sum / global_count
        global_emb_mean = global_emb_sum / global_count
        logger.info("SitePrototypes: %d sites, %d windows, K=%d", n_sites, global_count, K)

        # Center + normalize
        plc = proto_logits - global_logit_mean[None, None, :]
        proto_logits_normed = plc / np.maximum(np.linalg.norm(plc, axis=2, keepdims=True), 1e-8)

        pec = proto_emb - global_emb_mean[None, None, :]
        proto_emb_normed = pec / np.maximum(np.linalg.norm(pec, axis=2, keepdims=True), 1e-8)

        # Species × site-state affinity
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

        flat_protos = proto_logits_normed.reshape(-1, logit_dim)
        affinity = sp_normed @ flat_protos.T
        affinity[~has_sp] = 0
        affinity_max = affinity.reshape(n_species, n_sites, K).max(axis=2)

        # Save
        site_counts = [per_site_info.get(s, {}).get("n_windows", 0) for s in sites]
        np.savez_compressed(
            self.output()[0].path,
            proto_logits=proto_logits, proto_logits_normed=proto_logits_normed.astype(np.float32),
            proto_emb=proto_emb, proto_emb_normed=proto_emb_normed.astype(np.float32),
            prototypes_normed=flat_protos.astype(np.float32),
            proto_site=np.repeat(np.array(sites), K),
            proto_cluster_id=np.tile(np.arange(K, dtype=np.int32), n_sites),
            species_site_affinity=affinity.astype(np.float32),
            species_site_affinity_max=affinity_max.astype(np.float32),
            sites=np.array(sites), site_counts=np.array(site_counts, dtype=np.int64),
            global_soundscape_mean=global_logit_mean.astype(np.float32),
            global_emb_mean=global_emb_mean.astype(np.float32),
            species_list=np.array(species_list), n_prototypes=K,
        )

        summary = {
            "n_sites": n_sites, "total_windows": global_count,
            "n_prototypes_per_site": K, "total_prototypes": n_sites * K,
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
