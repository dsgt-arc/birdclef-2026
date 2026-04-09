"""Multi-prototype species profiles: NMF on logits, k-means on embeddings.

Each species is its own Luigi task → parallel execution, idempotent restarts.
MultiPrototypeProfiles aggregates and evaluates.

Output per species: artifacts/prototypes/species/{species}.npz
Aggregated: artifacts/multi_prototypes.npz + multi_prototypes_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import luigi
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.partition import PartitionFeatures
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    list_focal_species,
    load_focal_species,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

N_PROTOTYPES = 4
MIN_WINDOWS = 20


class SingleSpeciesPrototype(AnalysisTask):
    """NMF + k-means decomposition for one species."""

    species = luigi.Parameter()

    def requires(self):
        return PartitionFeatures(**self._params())

    def output(self):
        return self.artifact(f"prototypes/species/{self.species}.npz")

    def run(self):
        self.ensure_parent(self.output())
        K = N_PROTOTYPES

        df = load_focal_species(self.species, self.features_dir)
        logits = logits_to_numpy(df).astype(np.float64)
        emb = embeddings_to_numpy(df).astype(np.float64)
        n_windows = len(df)

        info = {"n_windows": n_windows, "n_prototypes": K}

        if n_windows < MIN_WINDOWS:
            H = np.tile(logits.mean(axis=0), (K, 1))
            centers = np.tile(emb.mean(axis=0), (K, 1))
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
                logger.warning("%s: NMF failed (%s), using mean", self.species, e)
                H = np.tile(logits.mean(axis=0), (k_actual, 1))

            if H.shape[0] < K:
                H = np.vstack([H, np.tile(H.mean(axis=0), (K - H.shape[0], 1))])

            # K-means on embeddings
            try:
                km = KMeans(n_clusters=k_actual, random_state=42, n_init=3)
                km.fit(emb)
                centers = km.cluster_centers_
                info["kmeans_inertia"] = round(float(km.inertia_), 4)
            except Exception as e:
                logger.warning("%s: k-means failed (%s), using mean", self.species, e)
                centers = np.tile(emb.mean(axis=0), (k_actual, 1))

            if centers.shape[0] < K:
                centers = np.vstack([centers, np.tile(centers.mean(axis=0), (K - centers.shape[0], 1))])

            info["method"] = "nmf+kmeans"

        np.savez(
            self.output().path,
            proto_logits=H.astype(np.float32),
            proto_emb=centers.astype(np.float32),
            logit_sum=logits.sum(axis=0),
            emb_sum=emb.sum(axis=0),
            n_windows=n_windows,
            info=json.dumps(info),
        )


class MultiPrototypeProfiles(AnalysisTask):
    """Aggregate per-species prototypes, evaluate on labeled soundscapes."""

    def requires(self):
        species_list = build_species_list(self.artifacts_dir)
        available = set(list_focal_species(self.features_dir))
        p = self._params()
        return {
            sp: SingleSpeciesPrototype(species=sp, **p)
            for sp in species_list if sp in available
        }

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
        K = N_PROTOTYPES

        logit_dim = emb_dim = None
        global_logit_sum = global_emb_sum = None
        global_count = 0
        per_species_info = {}
        proto_logits = proto_emb = None
        has_data = np.zeros(n_species, dtype=bool)

        for si, sp in enumerate(species_list):
            path = Path(self.artifacts_dir) / "prototypes" / "species" / f"{sp}.npz"
            if not path.exists():
                per_species_info[sp] = {"n_windows": 0, "n_prototypes": 0, "method": "none"}
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
                proto_logits = np.zeros((n_species, K, logit_dim), dtype=np.float32)
                proto_emb = np.zeros((n_species, K, emb_dim), dtype=np.float32)

            global_logit_sum += data["logit_sum"].astype(np.float64)
            global_emb_sum += data["emb_sum"].astype(np.float64)
            global_count += n_windows
            proto_logits[si] = H[:K]
            proto_emb[si] = centers[:K]
            has_data[si] = True
            per_species_info[sp] = info

        global_logit_mean = global_logit_sum / global_count
        global_emb_mean = global_emb_sum / global_count
        logger.info("MultiPrototype: %d species, %d windows, K=%d",
                     has_data.sum(), global_count, K)

        # Center + normalize
        plc = proto_logits - global_logit_mean[None, None, :]
        proto_logits_normed = plc / np.maximum(np.linalg.norm(plc, axis=2, keepdims=True), 1e-8)

        pec = proto_emb - global_emb_mean[None, None, :]
        proto_emb_normed = pec / np.maximum(np.linalg.norm(pec, axis=2, keepdims=True), 1e-8)

        # Evaluate
        logger.info("MultiPrototype: evaluating")
        sc_df = load_labeled(self.features_dir)
        sc_Y = build_binary_label_matrix(sc_df, species_list)
        sc_ln = normalize(logits_to_numpy(sc_df).astype(np.float64) - global_logit_mean[None, :], norm="l2")
        sc_en = normalize(embeddings_to_numpy(sc_df).astype(np.float64) - global_emb_mean[None, :], norm="l2")

        logit_scores = np.einsum("wp,skp->wsk", sc_ln, proto_logits_normed).max(axis=2)
        emb_scores = np.einsum("we,ske->wsk", sc_en, proto_emb_normed).max(axis=2)
        single_l_scores = sc_ln @ normalize(plc.mean(axis=1), norm="l2").T
        single_e_scores = sc_en @ normalize(pec.mean(axis=1), norm="l2").T

        eval_results = {}
        for si in range(n_species):
            sp = species_list[si]
            if not has_data[si] or sc_Y[:, si].sum() < 2:
                continue
            y = sc_Y[:, si]
            eval_results[sp] = {
                "auroc_logit_multi": round(float(roc_auc_score(y, logit_scores[:, si])), 4),
                "auroc_logit_single": round(float(roc_auc_score(y, single_l_scores[:, si])), 4),
                "auroc_emb_multi": round(float(roc_auc_score(y, emb_scores[:, si])), 4),
                "auroc_emb_single": round(float(roc_auc_score(y, single_e_scores[:, si])), 4),
            }

        for space in ["logit", "emb"]:
            vm = [v[f"auroc_{space}_multi"] for v in eval_results.values()]
            vs = [v[f"auroc_{space}_single"] for v in eval_results.values()]
            ni = sum(1 for v in eval_results.values() if v[f"auroc_{space}_multi"] > v[f"auroc_{space}_single"])
            logger.info("  %s: multi=%.4f single=%.4f, %d/%d improved",
                         space, np.mean(vm), np.mean(vs), ni, len(eval_results))

        # Save
        sp_count = np.array([per_species_info.get(sp, {}).get("n_windows", 0)
                             for sp in species_list], dtype=np.int64)
        np.savez_compressed(
            self.output()[0].path,
            proto_logits=proto_logits, proto_logits_normed=proto_logits_normed.astype(np.float32),
            proto_emb=proto_emb, proto_emb_normed=proto_emb_normed.astype(np.float32),
            prototypes_normed=proto_logits_normed.reshape(-1, logit_dim).astype(np.float32),
            proto_species_idx=np.repeat(np.arange(n_species, dtype=np.int32), K),
            proto_cluster_id=np.tile(np.arange(K, dtype=np.int32), n_species),
            species_list=np.array(species_list), species_count=sp_count,
            global_focal_mean=global_logit_mean.astype(np.float32),
            global_emb_mean=global_emb_mean.astype(np.float32),
            n_prototypes=K,
        )

        summary = {
            "n_species": n_species, "n_species_with_data": int(has_data.sum()),
            "n_prototypes_per_species": K,
            "total_prototypes": int(has_data.sum()) * K,
            "decomposition": {"logits": "NMF", "embeddings": "k-means"},
            "eval": self._eval_summary(eval_results),
            "per_species": per_species_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MultiPrototype: done")

    @staticmethod
    def _eval_summary(eval_results):
        out = {"n_evaluated": len(eval_results), "per_species": eval_results}
        if eval_results:
            for space in ["logit", "emb"]:
                for kind in ["multi", "single"]:
                    key = f"auroc_{space}_{kind}"
                    vals = [v[key] for v in eval_results.values()]
                    out[f"{space}_{kind}_auroc"] = round(float(np.mean(vals)), 4)
        return out
