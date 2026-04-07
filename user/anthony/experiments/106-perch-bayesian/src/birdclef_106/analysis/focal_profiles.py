"""Focal species profiles: per-species Perch logit signatures from focal recordings.

For each of the 206 focal species, compute multiple views of the Perch logit
activation pattern:

1. **Mean profile**: average logit values (sensitive to outliers)
2. **Median profile**: robust central tendency (ignores rare activations)
3. **Activation frequency**: fraction of windows where each logit exceeds a
   threshold — captures the sparse structure ("logit X fires 73% of the time
   for species Y")
4. **Signature summary**: for each species, the top-k most frequently activated
   logits with their activation rate, mean value, and the Perch species name

The activation frequency approach is best for sparse matching: background noise
averages out, consistent activations stand out regardless of amplitude.

Also computes secondary-label co-occurrence matrix from focal metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    extract_site,
    load_focal_shards,
    load_labeled,
    load_species_mapping,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

# A logit is "active" if it exceeds this percentile of the global logit distribution.
# Computed from a subsample in pass 1.
ACTIVATION_PERCENTILE = 75


class FocalSpeciesProfiles(AnalysisTask):
    """Build per-species logit signatures from focal data, evaluate on soundscapes."""

    def output(self):
        return [
            self.artifact("focal_profiles.npz"),
            self.artifact("focal_profiles_metrics.json"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        n_species = len(species_list)

        # Load Perch label names for interpretable output
        perch_labels = self._load_perch_labels()

        # === Pass 1: compute global activation threshold from a subsample ===

        logger.info("FocalProfiles: pass 1 — computing global activation threshold")
        logit_sample = []
        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard)
            logit_sample.append(logits[::8])  # subsample every 8th row
            if i >= 3:
                break
        logit_sample = np.concatenate(logit_sample)
        activation_threshold = float(np.percentile(logit_sample, ACTIVATION_PERCENTILE))
        logger.info("  Activation threshold (p%d): %.4f", ACTIVATION_PERCENTILE, activation_threshold)
        del logit_sample

        # === Pass 2: accumulate per-species statistics ===

        logger.info("FocalProfiles: pass 2 — accumulating per-species profiles")
        logit_dim = None
        species_sum = None           # (n_species, logit_dim) — for mean
        species_sum_sq = None        # (n_species, logit_dim) — for variance
        species_active_count = None  # (n_species, logit_dim) — count of windows above threshold
        species_count = np.zeros(n_species, dtype=np.int64)
        cooccurrence = np.zeros((n_species, n_species), dtype=np.int64)

        # For median: collect per-species logit values (memory-intensive, do for top species only)
        # Instead, use the online approach: store sorted quantile sketches via running stats

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                species_sum = np.zeros((n_species, logit_dim), dtype=np.float64)
                species_sum_sq = np.zeros((n_species, logit_dim), dtype=np.float64)
                species_active_count = np.zeros((n_species, logit_dim), dtype=np.int64)

            primary_labels = shard["primary_label"].to_list()
            ground_truths = shard["ground_truth"].to_list()

            for j in range(len(shard)):
                pl = str(primary_labels[j]).strip()
                si = species_to_idx.get(pl)
                if si is None:
                    continue

                row = logits[j].astype(np.float64)
                species_sum[si] += row
                species_sum_sq[si] += row ** 2
                species_active_count[si] += (logits[j] > activation_threshold).astype(np.int64)
                species_count[si] += 1

                # Co-occurrence from secondary labels
                gt = str(ground_truths[j]).strip()
                if gt:
                    for sp in gt.split(";"):
                        sp = sp.strip()
                        sj = species_to_idx.get(sp)
                        if sj is not None and sj != si:
                            cooccurrence[si, sj] += 1

            if i % 4 == 0:
                logger.info("  focal shard %d done", i)

        # Compute profiles
        mask = species_count > 0
        mean_profiles = np.zeros_like(species_sum)
        std_profiles = np.zeros_like(species_sum)
        activation_freq = np.zeros_like(species_sum)

        mean_profiles[mask] = species_sum[mask] / species_count[mask, None]
        var = species_sum_sq[mask] / species_count[mask, None] - mean_profiles[mask] ** 2
        std_profiles[mask] = np.sqrt(np.maximum(var, 0))
        activation_freq[mask] = species_active_count[mask] / species_count[mask, None]

        n_with_data = int(mask.sum())
        logger.info("FocalProfiles: %d/%d species have focal data", n_with_data, n_species)

        # === Build per-species signature summaries ===

        top_k = 30
        signatures = {}
        for si in range(n_species):
            sp = species_list[si]
            if species_count[si] == 0:
                continue

            freq = activation_freq[si]
            mean_val = mean_profiles[si]
            std_val = std_profiles[si]

            # Top logits by activation frequency
            top_idx = np.argsort(-freq)[:top_k]
            top_logits = []
            for j in top_idx:
                if freq[j] < 0.05:  # skip logits that fire <5% of the time
                    break
                perch_name = perch_labels[j] if j < len(perch_labels) else f"logit_{j}"
                top_logits.append({
                    "logit_index": int(j),
                    "perch_species": perch_name,
                    "activation_freq": float(freq[j]),
                    "mean_value": float(mean_val[j]),
                    "std_value": float(std_val[j]),
                })

            # Sparsity: how many logits fire >10% of the time?
            n_active_10pct = int((freq > 0.10).sum())
            n_active_25pct = int((freq > 0.25).sum())
            n_active_50pct = int((freq > 0.50).sum())

            signatures[sp] = {
                "n_focal_windows": int(species_count[si]),
                "n_active_10pct": n_active_10pct,
                "n_active_25pct": n_active_25pct,
                "n_active_50pct": n_active_50pct,
                "top_logits": top_logits,
            }

        # === Evaluate on labeled soundscapes ===

        logger.info("FocalProfiles: evaluating on labeled soundscapes")
        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)

        active_species = [i for i in range(n_species) if species_count[i] > 0]

        # Three matching strategies:
        eval_results = {}
        for strategy_name, profiles in [
            ("mean", mean_profiles),
            ("activation_freq", activation_freq),
        ]:
            profiles_active = profiles[active_species]

            # Normalize for cosine similarity
            norms = np.linalg.norm(profiles_active, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            profiles_normed = profiles_active / norms

            sc_norms = np.linalg.norm(sc_logits, axis=1, keepdims=True)
            sc_norms = np.maximum(sc_norms, 1e-8)
            sc_normed = sc_logits / sc_norms

            sim_matrix = sc_normed @ profiles_normed.T  # (739, n_active)

            per_species_eval = {}
            for idx_in_active, si in enumerate(active_species):
                sp = species_list[si]
                y_true = sc_Y[:, si]
                n_pos = int(y_true.sum())
                if n_pos < 2:
                    per_species_eval[sp] = {"auroc": None, "ap": None, "n_pos": n_pos}
                    continue

                y_score = sim_matrix[:, idx_in_active]
                per_species_eval[sp] = {
                    "auroc": float(roc_auc_score(y_true, y_score)),
                    "ap": float(average_precision_score(y_true, y_score)),
                    "n_pos": n_pos,
                }

            aurocs = [v["auroc"] for v in per_species_eval.values() if v["auroc"] is not None]
            aps = [v["ap"] for v in per_species_eval.values() if v["ap"] is not None]

            eval_results[strategy_name] = {
                "per_species": per_species_eval,
                "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
                "macro_ap": float(np.mean(aps)) if aps else None,
                "n_evaluated": len(aurocs),
            }

            logger.info(
                "FocalProfiles [%s]: macro AUROC=%.3f, AP=%.3f (%d species)",
                strategy_name, eval_results[strategy_name]["macro_auroc"] or 0,
                eval_results[strategy_name]["macro_ap"] or 0,
                eval_results[strategy_name]["n_evaluated"],
            )

        # === Save ===

        np.savez_compressed(
            self.output()[0].path,
            mean_profiles=mean_profiles.astype(np.float32),
            std_profiles=std_profiles.astype(np.float32),
            activation_freq=activation_freq.astype(np.float32),
            species_count=species_count,
            cooccurrence=cooccurrence.astype(np.int32),
            species_list=species_list,
            activation_threshold=activation_threshold,
        )

        # Top co-occurring species pairs
        top_cooc = []
        for si in range(n_species):
            for sj in range(si + 1, n_species):
                c = int(cooccurrence[si, sj] + cooccurrence[sj, si])
                if c > 10:
                    top_cooc.append({
                        "species_a": species_list[si],
                        "species_b": species_list[sj],
                        "count": c,
                    })
        top_cooc.sort(key=lambda x: -x["count"])

        summary = {
            "evaluation": eval_results,
            "signatures": signatures,
            "n_species_with_focal": n_with_data,
            "activation_threshold": activation_threshold,
            "activation_percentile": ACTIVATION_PERCENTILE,
            "top_cooccurrences": top_cooc[:50],
        }

        with open(self.output()[1].path, "w") as f:
            json.dump(summary, f, indent=2)

    def _load_perch_labels(self) -> list[str]:
        """Load Perch v2 species names for interpretable output."""
        labels_path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not labels_path.exists():
            return []
        with open(labels_path) as f:
            return [line.strip() for line in f]
