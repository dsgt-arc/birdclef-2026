"""Focal species profiles: per-species Perch logit signatures from focal recordings.

Uses partitioned by_species/ parquets — loads one species at a time instead of
scanning all 16 shards (15 GB). Peak memory: ~120 MB per species.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.structure.partition import PartitionFeatures
from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    list_focal_species,
    load_focal_species,
    load_labeled,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

ACTIVATION_PERCENTILE = 75
TOP_K = 30
MIN_ACTIVATION_FREQ = 0.05


class FocalSpeciesProfiles(AnalysisTask):
    """Build per-species logit signatures from focal data, evaluate on soundscapes."""

    def requires(self):
        return PartitionFeatures(**self._params())

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
        available = set(list_focal_species(self.features_dir))
        perch_labels = self._load_perch_labels()

        # === Pass 1: sample a few species for global activation threshold ===
        sample_species = sorted(available)[:10]
        sample_logits = []
        for sp in sample_species:
            df = load_focal_species(sp, self.features_dir)
            sample_logits.append(logits_to_numpy(df)[::8])
        sample = np.concatenate(sample_logits)
        threshold = float(np.percentile(sample, ACTIVATION_PERCENTILE))
        logger.info("FocalProfiles: threshold (p%d): %.4f from %d samples",
                     ACTIVATION_PERCENTILE, threshold, len(sample))
        del sample, sample_logits

        # === Pass 2: iterate per species ===
        logger.info("FocalProfiles: computing profiles for %d species", len(available))
        logit_dim = None
        sp_sum = sp_sum_sq = sp_active = None
        sp_count = np.zeros(n_species, dtype=np.int64)
        cooc = np.zeros((n_species, n_species), dtype=np.int64)

        for sp_i, sp in enumerate(species_list):
            if sp not in available:
                continue

            df = load_focal_species(sp, self.features_dir)
            logits = logits_to_numpy(df)

            if logit_dim is None:
                logit_dim = logits.shape[1]
                sp_sum = np.zeros((n_species, logit_dim), dtype=np.float64)
                sp_sum_sq = np.zeros((n_species, logit_dim), dtype=np.float64)
                sp_active = np.zeros((n_species, logit_dim), dtype=np.int64)

            logits_f64 = logits.astype(np.float64)
            sp_sum[sp_i] = logits_f64.sum(axis=0)
            sp_sum_sq[sp_i] = (logits_f64 ** 2).sum(axis=0)
            sp_active[sp_i] = (logits > threshold).sum(axis=0)
            sp_count[sp_i] = len(df)

            # Co-occurrence from ground_truth
            if "ground_truth" in df.columns:
                for gt in df["ground_truth"].to_list():
                    if not gt:
                        continue
                    for co_sp in str(gt).split(";"):
                        sj = species_to_idx.get(co_sp.strip())
                        if sj is not None and sj != sp_i:
                            cooc[sp_i, sj] += 1

            if (sp_i + 1) % 50 == 0:
                logger.info("  %d/%d species done", sp_i + 1, n_species)

            del df, logits, logits_f64

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
        if cooc_rows:
            cooc_df = pl.DataFrame(cooc_rows).sort("count", descending=True)
        else:
            cooc_df = pl.DataFrame({"species_a": [], "species_b": [], "count": []})
        cooc_df.write_parquet(self.output()[1].path)

        # === Raw profiles npz ===
        np.savez_compressed(
            self.output()[2].path,
            mean_profiles=mean_prof.astype(np.float32),
            std_profiles=std_prof.astype(np.float32),
            activation_freq=act_freq.astype(np.float32),
            species_count=sp_count,
            species_list=species_list,
            activation_threshold=threshold,
        )

        aurocs_mean = [r["auroc_mean"] for r in rows if r["auroc_mean"] is not None]
        aurocs_act = [r["auroc_activation"] for r in rows if r["auroc_activation"] is not None]
        logger.info(
            "FocalProfiles: mean AUROC=%.3f, activation AUROC=%.3f (%d species)",
            np.mean(aurocs_mean) if aurocs_mean else 0,
            np.mean(aurocs_act) if aurocs_act else 0,
            len(aurocs_mean),
        )

    def _load_perch_labels(self) -> list[str]:
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
