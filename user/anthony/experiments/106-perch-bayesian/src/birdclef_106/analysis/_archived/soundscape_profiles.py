"""Soundscape profiles: per-site Perch logit activation signatures.

Mirrors focal_profiles.py but groups by site instead of species. Shows what
Perch detects at each of the 23 monitoring sites — no labels needed.

Output parquet has one row per site with nested signature list:

    df = pl.read_parquet("artifacts/soundscape_signatures.parquet")
    df.sort("n_windows", descending=True)

    # What Perch species characterize site S22?
    df.filter(pl.col("site") == "S22").explode("signature").unnest("signature")

    # Which Perch species appear at many sites?
    df.explode("signature").unnest("signature") \\
        .group_by("perch_species").agg(pl.col("site").n_unique())

Also computes diel profiles (4 time periods) per site, showing how
activations shift from night → dawn → day → dusk.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import polars as pl

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    extract_site,
    load_labeled,
    load_unlabeled_shards,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

ACTIVATION_PERCENTILE = 75
TOP_K = 50
MIN_ACTIVATION_FREQ = 0.05

# Diel periods based on hour (Pantanal tropical, ~6am sunrise, ~6pm sunset)
DIEL_PERIODS = {
    "night": (0, 5),    # 00:00 - 04:59
    "dawn": (5, 8),     # 05:00 - 07:59
    "day": (8, 17),     # 08:00 - 16:59
    "dusk": (17, 21),   # 17:00 - 20:59
    # 21:00 - 23:59 → night (handled by modular logic)
}

# Filename: BC2026_Train_{id}_{site}_{date}_{time}.ogg
# Time field: HHMMSS
_TIME_RE = re.compile(r"_(\d{6})\.")


def _extract_hour(filename: str) -> int | None:
    """Extract hour from soundscape filename (HHMMSS before .ogg)."""
    m = _TIME_RE.search(filename)
    if m:
        return int(m.group(1)[:2])
    return None


def _hour_to_period(hour: int) -> str:
    if hour >= 21 or hour < 5:
        return "night"
    if hour < 8:
        return "dawn"
    if hour < 17:
        return "day"
    return "dusk"


SIG_DTYPE = pl.List(pl.Struct({
    "logit_index": pl.Int32,
    "perch_species": pl.Utf8,
    "activation_freq": pl.Float32,
    "mean_value": pl.Float32,
    "std_value": pl.Float32,
}))


class SoundscapeProfiles(AnalysisTask):
    """Per-site and per-diel-period Perch logit activation profiles."""

    def output(self):
        return [
            self.artifact("soundscape_signatures.parquet"),
            self.artifact("soundscape_diel_signatures.parquet"),
            self.artifact("soundscape_profiles_raw.npz"),
        ]

    def run(self):
        for t in self.output():
            self.ensure_parent(t)

        perch_labels = self._load_perch_labels()

        # === Pass 1: global activation threshold (reuse from focal or compute) ===

        logger.info("SoundscapeProfiles: pass 1 — activation threshold")
        sample = []
        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            sample.append(logits_to_numpy(shard)[::8])
            if i >= 3:
                break
        sample = np.concatenate(sample)
        threshold = float(np.percentile(sample, ACTIVATION_PERCENTILE))
        logger.info("  threshold (p%d): %.4f", ACTIVATION_PERCENTILE, threshold)
        del sample

        # === Pass 2: accumulate per-site and per-site-period stats ===

        logger.info("SoundscapeProfiles: pass 2 — accumulating by site and diel period")
        logit_dim = None

        # Per-site accumulators
        site_sum: dict[str, np.ndarray] = {}
        site_sum_sq: dict[str, np.ndarray] = {}
        site_active: dict[str, np.ndarray] = {}
        site_count: dict[str, int] = {}

        # Per-site-period accumulators
        sp_sum: dict[tuple[str, str], np.ndarray] = {}
        sp_active: dict[tuple[str, str], np.ndarray] = {}
        sp_count: dict[tuple[str, str], int] = {}

        # Also process labeled soundscapes
        all_shards = list(load_unlabeled_shards(self.features_dir))
        labeled = load_labeled(self.features_dir)
        all_shards.append(labeled)

        for i, shard in enumerate(all_shards):
            logits = logits_to_numpy(shard)
            if logit_dim is None:
                logit_dim = logits.shape[1]

            filenames = shard["filename"].to_list()

            for j in range(len(shard)):
                site = extract_site(filenames[j])
                hour = _extract_hour(filenames[j])
                period = _hour_to_period(hour) if hour is not None else "unknown"
                row = logits[j].astype(np.float64)
                active = (logits[j] > threshold).astype(np.int64)

                # Per-site
                if site not in site_sum:
                    site_sum[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_sum_sq[site] = np.zeros(logit_dim, dtype=np.float64)
                    site_active[site] = np.zeros(logit_dim, dtype=np.int64)
                    site_count[site] = 0
                site_sum[site] += row
                site_sum_sq[site] += row ** 2
                site_active[site] += active
                site_count[site] += 1

                # Per-site-period
                key = (site, period)
                if key not in sp_sum:
                    sp_sum[key] = np.zeros(logit_dim, dtype=np.float64)
                    sp_active[key] = np.zeros(logit_dim, dtype=np.int64)
                    sp_count[key] = 0
                sp_sum[key] += row
                sp_active[key] += active
                sp_count[key] += 1

            if i % 4 == 0:
                logger.info("  shard %d done", i)

        logger.info("SoundscapeProfiles: %d sites found", len(site_count))

        # === Build per-site signatures ===

        site_rows = []
        all_sites = sorted(site_count.keys())
        # For raw npz
        n_sites = len(all_sites)
        mean_profiles = np.zeros((n_sites, logit_dim), dtype=np.float32)
        act_freq_profiles = np.zeros((n_sites, logit_dim), dtype=np.float32)

        for si, site in enumerate(all_sites):
            n = site_count[site]
            mean_prof = site_sum[site] / n
            var = site_sum_sq[site] / n - mean_prof ** 2
            std_prof = np.sqrt(np.maximum(var, 0))
            act_freq = site_active[site] / n

            mean_profiles[si] = mean_prof.astype(np.float32)
            act_freq_profiles[si] = act_freq.astype(np.float32)

            sig = self._build_signature(act_freq, mean_prof, std_prof, perch_labels)

            site_rows.append({
                "site": site,
                "n_windows": n,
                "n_active_10pct": int((act_freq > 0.10).sum()),
                "n_active_25pct": int((act_freq > 0.25).sum()),
                "n_active_50pct": int((act_freq > 0.50).sum()),
                "signature": sig,
            })

        site_df = pl.DataFrame(site_rows, schema_overrides={"signature": SIG_DTYPE})
        site_df.write_parquet(self.output()[0].path)
        logger.info("SoundscapeProfiles: wrote %d sites", len(site_rows))

        # === Build per-site-period (diel) signatures ===

        diel_rows = []
        for (site, period), n in sorted(sp_count.items()):
            mean_prof = sp_sum[(site, period)] / n
            act_freq = sp_active[(site, period)] / n
            std_prof = np.zeros(logit_dim)  # skip std for diel to save memory

            sig = self._build_signature(act_freq, mean_prof, std_prof, perch_labels)

            diel_rows.append({
                "site": site,
                "period": period,
                "n_windows": n,
                "n_active_10pct": int((act_freq > 0.10).sum()),
                "n_active_50pct": int((act_freq > 0.50).sum()),
                "signature": sig,
            })

        diel_df = pl.DataFrame(diel_rows, schema_overrides={"signature": SIG_DTYPE})
        diel_df.write_parquet(self.output()[1].path)
        logger.info("SoundscapeProfiles: wrote %d site-period rows", len(diel_rows))

        # === Raw profiles npz ===

        np.savez_compressed(
            self.output()[2].path,
            mean_profiles=mean_profiles,
            activation_freq=act_freq_profiles,
            sites=all_sites,
            site_counts=np.array([site_count[s] for s in all_sites]),
            activation_threshold=threshold,
        )

    def _build_signature(self, act_freq, mean_prof, std_prof, perch_labels):
        """Build top-k signature list from activation frequency."""
        top_idx = np.argsort(-act_freq)[:TOP_K]
        sig = []
        for j in top_idx:
            if act_freq[j] < MIN_ACTIVATION_FREQ:
                break
            pname = perch_labels[j] if j < len(perch_labels) else f"logit_{j}"
            sig.append({
                "logit_index": int(j),
                "perch_species": pname,
                "activation_freq": round(float(act_freq[j]), 4),
                "mean_value": round(float(mean_prof[j]), 4),
                "std_value": round(float(std_prof[j]), 4),
            })
        return sig

    def _load_perch_labels(self) -> list[str]:
        path = Path(self.artifacts_dir) / "perch_v2_labels.csv"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f]
