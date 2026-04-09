"""Multi-prototype species profiles: NMF on logits, k-means on embeddings.

For each species, decompose focal windows into K prototype patterns:
  - P-space (logits): NMF — non-negative, parts-based, additive
  - E-space (embeddings): k-means — simple, fast, Voronoi centroids

Fixed K per species gives a clean (S, K, P) tensor — no ragged arrays.
Both spaces run the same pipeline, just different decomposition method.

Depends on PartitionFeatures for per-species parquets.

Outputs:
    multi_prototypes.npz — prototype tensors, global means, metadata
    multi_prototypes_summary.json — per-species cluster info, evaluation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.partition import PartitionFeatures
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

# Number of prototypes per species (fixed K for clean tensor shape)
N_PROTOTYPES = 4

# Minimum windows to attempt decomposition (fall back to mean if fewer)
MIN_WINDOWS = 20


class MultiPrototypeProfiles(AnalysisTask):
    """Build multi-prototype species profiles via NMF (logits) + k-means (embeddings)."""

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
        K = N_PROTOTYPES

        # =====================================================================
        # Per-species decomposition
        # =====================================================================
        logger.info("MultiPrototype: decomposing %d species (K=%d)", n_species, K)

        logit_dim = None
        emb_dim = None

        # Accumulators for global means
        global_logit_sum = None
        global_emb_sum = None
        global_count = 0

        # Per-species results (filled in loop below)
        sp_logit_protos = {}   # species -> (K, P) or None
        sp_emb_protos = {}     # species -> (K, E) or None
        sp_counts = {}
        per_species_info = {}

        for idx, sp in enumerate(species_list):
            if sp not in available:
                per_species_info[sp] = {
                    "n_windows": 0, "n_prototypes": 0, "method": "none",
                }
                continue

            df = load_focal_species(sp, self.features_dir)
            logits = logits_to_numpy(df).astype(np.float64)
            emb = embeddings_to_numpy(df).astype(np.float64)
            n_windows = len(df)
            sp_counts[sp] = n_windows

            if logit_dim is None:
                logit_dim = logits.shape[1]
                emb_dim = emb.shape[1]
                global_logit_sum = np.zeros(logit_dim, dtype=np.float64)
                global_emb_sum = np.zeros(emb_dim, dtype=np.float64)

            global_logit_sum += logits.sum(axis=0)
            global_emb_sum += emb.sum(axis=0)
            global_count += n_windows

            if n_windows < MIN_WINDOWS:
                # Fall back to single mean (replicated K times)
                sp_logit_protos[sp] = np.tile(logits.mean(axis=0), (K, 1))
                sp_emb_protos[sp] = np.tile(emb.mean(axis=0), (K, 1))
                per_species_info[sp] = {
                    "n_windows": n_windows, "n_prototypes": K,
                    "method": "mean_fallback",
                }
                continue

            # NMF on logits (non-negative → parts-based prototypes)
            # Clip small negatives from numerical noise
            logits_nn = np.maximum(logits, 0).astype(np.float64)
            k_actual = min(K, n_windows - 1)

            try:
                nmf = NMF(n_components=k_actual, init="nndsvda", max_iter=300,
                          random_state=42)
                W = nmf.fit_transform(logits_nn)
                H = nmf.components_  # (k_actual, P)
                nmf_error = nmf.reconstruction_err_
            except Exception as e:
                logger.warning("  %s: NMF failed (%s), using mean", sp, e)
                H = np.tile(logits.mean(axis=0), (k_actual, 1))
                nmf_error = -1.0

            # Pad to K if k_actual < K
            if H.shape[0] < K:
                pad = np.tile(H.mean(axis=0), (K - H.shape[0], 1))
                H = np.vstack([H, pad])
            sp_logit_protos[sp] = H

            # K-means on embeddings
            try:
                km = KMeans(n_clusters=k_actual, random_state=42, n_init=3)
                km.fit(emb)
                centers = km.cluster_centers_  # (k_actual, E)
                km_inertia = km.inertia_
            except Exception as e:
                logger.warning("  %s: k-means failed (%s), using mean", sp, e)
                centers = np.tile(emb.mean(axis=0), (k_actual, 1))
                km_inertia = -1.0

            if centers.shape[0] < K:
                pad = np.tile(centers.mean(axis=0), (K - centers.shape[0], 1))
                centers = np.vstack([centers, pad])
            sp_emb_protos[sp] = centers

            per_species_info[sp] = {
                "n_windows": n_windows,
                "n_prototypes": K,
                "method": "nmf+kmeans",
                "nmf_reconstruction_error": round(float(nmf_error), 4) if nmf_error >= 0 else None,
                "kmeans_inertia": round(float(km_inertia), 4) if km_inertia >= 0 else None,
            }

            if idx % 20 == 0:
                logger.info("  species %d/%d (%s): %d windows", idx, n_species, sp, n_windows)

        # Global means
        global_logit_mean = global_logit_sum / global_count
        global_emb_mean = global_emb_sum / global_count

        logger.info("MultiPrototype: %d species with data, %d total windows",
                     len(sp_counts), global_count)

        # =====================================================================
        # Build prototype tensors (S, K, P) and (S, K, E)
        # =====================================================================

        proto_logits = np.zeros((n_species, K, logit_dim), dtype=np.float32)
        proto_emb = np.zeros((n_species, K, emb_dim), dtype=np.float32)
        has_data = np.zeros(n_species, dtype=bool)

        for si, sp in enumerate(species_list):
            if sp in sp_logit_protos:
                proto_logits[si] = sp_logit_protos[sp].astype(np.float32)
                proto_emb[si] = sp_emb_protos[sp].astype(np.float32)
                has_data[si] = True

        # Center and normalize logit prototypes for cosine scoring
        proto_logits_centered = proto_logits - global_logit_mean[None, None, :]
        norms = np.maximum(np.linalg.norm(proto_logits_centered, axis=2, keepdims=True), 1e-8)
        proto_logits_normed = proto_logits_centered / norms

        # Normalize embedding prototypes
        proto_emb_centered = proto_emb - global_emb_mean[None, None, :]
        norms = np.maximum(np.linalg.norm(proto_emb_centered, axis=2, keepdims=True), 1e-8)
        proto_emb_normed = proto_emb_centered / norms

        # =====================================================================
        # Evaluate on labeled soundscapes
        # =====================================================================
        logger.info("MultiPrototype: evaluating")

        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_emb = embeddings_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)

        # Logit-space scoring: max over K prototypes per species
        sc_logit_centered = sc_logits - global_logit_mean[None, :]
        sc_logit_normed = normalize(sc_logit_centered, norm="l2")
        # (N_sc, S, K) via einsum, then max over K
        logit_sims = np.einsum("wp,skp->wsk", sc_logit_normed, proto_logits_normed)
        logit_scores = logit_sims.max(axis=2)  # (N_sc, S)

        # Embedding-space scoring
        sc_emb_centered = sc_emb - global_emb_mean[None, :]
        sc_emb_normed = normalize(sc_emb_centered, norm="l2")
        emb_sims = np.einsum("we,ske->wsk", sc_emb_normed, proto_emb_normed)
        emb_scores = emb_sims.max(axis=2)  # (N_sc, S)

        # Single-mean baseline (logits)
        single_logit = proto_logits_centered.mean(axis=1)  # (S, P) — mean of K prototypes
        single_logit_normed = normalize(single_logit, norm="l2")
        single_logit_scores = sc_logit_normed @ single_logit_normed.T  # (N_sc, S)

        # Single-mean baseline (embeddings)
        single_emb = proto_emb_centered.mean(axis=1)
        single_emb_normed = normalize(single_emb, norm="l2")
        single_emb_scores = sc_emb_normed @ single_emb_normed.T

        # Per-species metrics
        eval_results = {}
        for si in range(n_species):
            sp = species_list[si]
            if not has_data[si] or sc_Y[:, si].sum() < 2:
                continue
            y = sc_Y[:, si]

            eval_results[sp] = {
                "auroc_logit_multi": round(float(roc_auc_score(y, logit_scores[:, si])), 4),
                "auroc_logit_single": round(float(roc_auc_score(y, single_logit_scores[:, si])), 4),
                "auroc_emb_multi": round(float(roc_auc_score(y, emb_scores[:, si])), 4),
                "auroc_emb_single": round(float(roc_auc_score(y, single_emb_scores[:, si])), 4),
            }

        for space in ["logit", "emb"]:
            multi_key = f"auroc_{space}_multi"
            single_key = f"auroc_{space}_single"
            multi_vals = [v[multi_key] for v in eval_results.values()]
            single_vals = [v[single_key] for v in eval_results.values()]
            n_improved = sum(1 for v in eval_results.values() if v[multi_key] > v[single_key])
            logger.info("  %s: multi=%.4f single=%.4f, %d/%d improved",
                         space, np.mean(multi_vals), np.mean(single_vals),
                         n_improved, len(eval_results))

        # =====================================================================
        # Save
        # =====================================================================
        sp_count_arr = np.array([sp_counts.get(sp, 0) for sp in species_list], dtype=np.int64)

        np.savez_compressed(
            self.output()[0].path,
            # (S, K, P) and (S, K, E) prototype tensors
            proto_logits=proto_logits,
            proto_logits_normed=proto_logits_normed.astype(np.float32),
            proto_emb=proto_emb,
            proto_emb_normed=proto_emb_normed.astype(np.float32),
            # Flat versions for PrototypeRidge compatibility
            prototypes_normed=proto_logits_normed.reshape(-1, logit_dim).astype(np.float32),
            proto_species_idx=np.repeat(np.arange(n_species, dtype=np.int32), K),
            proto_cluster_id=np.tile(np.arange(K, dtype=np.int32), n_species),
            # Metadata
            species_list=np.array(species_list),
            species_count=sp_count_arr,
            global_focal_mean=global_logit_mean.astype(np.float32),
            global_emb_mean=global_emb_mean.astype(np.float32),
            n_prototypes=K,
        )

        summary = {
            "n_species": n_species,
            "n_species_with_data": int(has_data.sum()),
            "n_prototypes_per_species": K,
            "total_prototypes": int(has_data.sum()) * K,
            "decomposition": {"logits": "NMF", "embeddings": "k-means"},
            "eval": {
                "n_evaluated": len(eval_results),
                "logit_multi_auroc": round(float(np.mean([v["auroc_logit_multi"] for v in eval_results.values()])), 4),
                "logit_single_auroc": round(float(np.mean([v["auroc_logit_single"] for v in eval_results.values()])), 4),
                "emb_multi_auroc": round(float(np.mean([v["auroc_emb_multi"] for v in eval_results.values()])), 4),
                "emb_single_auroc": round(float(np.mean([v["auroc_emb_single"] for v in eval_results.values()])), 4),
                "per_species": eval_results,
            },
            "per_species": per_species_info,
        }
        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MultiPrototype: done")
