"""Focal species profiles: per-species Perch logit signatures from focal recordings.

For each of the 206 focal species, compute the Perch logit activation pattern:

1. **Mean profile**: average logit values across all focal windows
2. **Activation frequency**: fraction of windows where each logit exceeds a
   threshold — captures the sparse structure

Output is a single parquet with nested signature lists, queryable via polars:

    df = pl.read_parquet("artifacts/focal_signatures.parquet")
    df.sort("auroc_activation", descending=True)

    # Explode to see which Perch species fire for each BirdCLEF species
    df.explode("signature").unnest("signature")

    # Which Perch species appear across many BirdCLEF signatures?
    df.explode("signature").unnest("signature") \\
        .group_by("perch_species").agg(pl.col("species").n_unique())

Also outputs a co-occurrence parquet from secondary labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    load_focal_shards,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

ACTIVATION_PERCENTILE = 75
TOP_K = 30
MIN_ACTIVATION_FREQ = 0.05


class FocalSpeciesProfiles(AnalysisTask):
    """Build per-species logit signatures from focal data, evaluate on soundscapes."""

    def output(self):
        return [
            self.artifact("focal_signatures.parquet"),
            self.artifact("focal_cooccurrence.parquet"),
            self.artifact("focal_profiles_raw.npz"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        species_list = build_species_list(self.artifacts_dir)
        species_to_idx = {s: i for i, s in enumerate(species_list)}
        n_species = len(species_list)
        perch_labels = self._load_perch_labels()

        # === Pass 1: global activation threshold ===

        logger.info("FocalProfiles: pass 1 — activation threshold")
        sample = []
        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            sample.append(logits_to_numpy(shard)[::8])
            if i >= 3:
                break
        sample = np.concatenate(sample)
        threshold = float(np.percentile(sample, ACTIVATION_PERCENTILE))
        logger.info("  threshold (p%d): %.4f", ACTIVATION_PERCENTILE, threshold)
        del sample

        # === Pass 2: accumulate per-species stats ===

        logger.info("FocalProfiles: pass 2 — accumulating")
        logit_dim = None
        sp_sum = None
        sp_sum_sq = None
        sp_active = None
        sp_count = np.zeros(n_species, dtype=np.int64)
        cooc = np.zeros((n_species, n_species), dtype=np.int64)

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            logits = logits_to_numpy(shard)
            if logit_dim is None:
                logit_dim = logits.shape[1]
                sp_sum = np.zeros((n_species, logit_dim), dtype=np.float64)
                sp_sum_sq = np.zeros((n_species, logit_dim), dtype=np.float64)
                sp_active = np.zeros((n_species, logit_dim), dtype=np.int64)

            pls = shard["primary_label"].to_list()
            gts = shard["ground_truth"].to_list()

            for j in range(len(shard)):
                si = species_to_idx.get(str(pls[j]).strip())
                if si is None:
                    continue
                row = logits[j].astype(np.float64)
                sp_sum[si] += row
                sp_sum_sq[si] += row ** 2
                sp_active[si] += (logits[j] > threshold).astype(np.int64)
                sp_count[si] += 1

                gt = str(gts[j]).strip()
                if gt:
                    for sp in gt.split(";"):
                        sj = species_to_idx.get(sp.strip())
                        if sj is not None and sj != si:
                            cooc[si, sj] += 1

            if i % 4 == 0:
                logger.info("  shard %d done", i)

        # Compute profiles
        has_data = sp_count > 0
        mean_prof = np.zeros_like(sp_sum)
        std_prof = np.zeros_like(sp_sum)
        act_freq = np.zeros_like(sp_sum)

        mean_prof[has_data] = sp_sum[has_data] / sp_count[has_data, None]
        var = sp_sum_sq[has_data] / sp_count[has_data, None] - mean_prof[has_data] ** 2
        std_prof[has_data] = np.sqrt(np.maximum(var, 0))
        act_freq[has_data] = sp_active[has_data] / sp_count[has_data, None]

        logger.info("FocalProfiles: %d/%d species have focal data", has_data.sum(), n_species)

        # === Evaluate on soundscapes ===

        logger.info("FocalProfiles: evaluating on soundscapes")
        sc_df = load_labeled(self.features_dir)
        sc_logits = logits_to_numpy(sc_df).astype(np.float64)
        sc_Y = build_binary_label_matrix(sc_df, species_list)

        active_idx = [i for i in range(n_species) if sp_count[i] > 0]

        eval_results = {}
        for strat_name, profiles in [("mean", mean_prof), ("activation", act_freq)]:
            prof = profiles[active_idx]
            norms = np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-8)
            prof_n = prof / norms
            sc_n = sc_logits / np.maximum(np.linalg.norm(sc_logits, axis=1, keepdims=True), 1e-8)
            sims = sc_n @ prof_n.T

            per_sp = {}
            for ii, si in enumerate(active_idx):
                y = sc_Y[:, si]
                n_pos = int(y.sum())
                if n_pos < 2:
                    per_sp[species_list[si]] = None
                    continue
                per_sp[species_list[si]] = {
                    "auroc": float(roc_auc_score(y, sims[:, ii])),
                    "ap": float(average_precision_score(y, sims[:, ii])),
                }
            eval_results[strat_name] = per_sp

        # === Build polars output ===

        rows = []
        for si in range(n_species):
            sp = species_list[si]
            if sp_count[si] == 0:
                continue

            freq = act_freq[si]
            mv = mean_prof[si]
            sv = std_prof[si]

            # Top logits by activation frequency
            top_idx = np.argsort(-freq)[:TOP_K]
            sig = []
            for j in top_idx:
                if freq[j] < MIN_ACTIVATION_FREQ:
                    break
                pname = perch_labels[j] if j < len(perch_labels) else f"logit_{j}"
                sig.append({
                    "logit_index": int(j),
                    "perch_species": pname,
                    "activation_freq": round(float(freq[j]), 4),
                    "mean_value": round(float(mv[j]), 4),
                    "std_value": round(float(sv[j]), 4),
                })

            mean_eval = eval_results["mean"].get(sp)
            act_eval = eval_results["activation"].get(sp)

            rows.append({
                "species": sp,
                "n_focal_windows": int(sp_count[si]),
                "n_active_10pct": int((freq > 0.10).sum()),
                "n_active_25pct": int((freq > 0.25).sum()),
                "n_active_50pct": int((freq > 0.50).sum()),
                "auroc_mean": mean_eval["auroc"] if mean_eval else None,
                "auroc_activation": act_eval["auroc"] if act_eval else None,
                "ap_mean": mean_eval["ap"] if mean_eval else None,
                "ap_activation": act_eval["ap"] if act_eval else None,
                "signature": sig,
            })

        # Build DataFrame with explicit signature struct type to handle empty lists.
        # Polars can't infer List(Struct) from an empty list.
        sig_dtype = pl.List(pl.Struct({
            "logit_index": pl.Int32,
            "perch_species": pl.Utf8,
            "activation_freq": pl.Float32,
            "mean_value": pl.Float32,
            "std_value": pl.Float32,
        }))
        sig_df = pl.DataFrame(rows, schema_overrides={"signature": sig_dtype})
        sig_df.write_parquet(self.output()[0].path)
        logger.info("FocalProfiles: wrote %d species to %s", len(rows), self.output()[0].path)

        # === Co-occurrence parquet ===

        cooc_rows = []
        for si in range(n_species):
            for sj in range(si + 1, n_species):
                c = int(cooc[si, sj] + cooc[sj, si])
                if c > 0:
                    cooc_rows.append({
                        "species_a": species_list[si],
                        "species_b": species_list[sj],
                        "count": c,
                    })
        cooc_df = pl.DataFrame(cooc_rows).sort("count", descending=True)
        cooc_df.write_parquet(self.output()[1].path)
        logger.info("FocalProfiles: %d co-occurrence pairs", len(cooc_rows))

        # === Raw profiles npz (for downstream ridge/matching) ===

        np.savez_compressed(
            self.output()[2].path,
            mean_profiles=mean_prof.astype(np.float32),
            std_profiles=std_prof.astype(np.float32),
            activation_freq=act_freq.astype(np.float32),
            species_count=sp_count,
            species_list=species_list,
            activation_threshold=threshold,
        )

        # Log summary
        aurocs_mean = [r["auroc_mean"] for r in rows if r["auroc_mean"] is not None]
        aurocs_act = [r["auroc_activation"] for r in rows if r["auroc_activation"] is not None]
        logger.info(
            "FocalProfiles: mean matching AUROC=%.3f, activation matching AUROC=%.3f (%d species)",
            np.mean(aurocs_mean) if aurocs_mean else 0,
            np.mean(aurocs_act) if aurocs_act else 0,
            len(aurocs_mean),
        )

    def _load_perch_labels(self) -> list[str]:
        """Load Perch v2 species names for interpretable output."""
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
