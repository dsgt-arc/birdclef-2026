"""KS-shift diagnostic between two embedding sets (ported from exp 106 domain_shift).

Default use: frozen vs adapted embeddings of the SAME labeled windows for one
encoder — "did adaptation move the representation, and how much?". The same code
serves the focal↔soundscape pull-together question by pointing it at a focal cache
vs a soundscape cache (per-dim KS over the two domains, before and after DAPT).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

logger = logging.getLogger("birdclef_301.ks_shift")


def _embeddings(parquet_path: str | Path) -> np.ndarray:
    df = pl.read_parquet(parquet_path)
    return np.stack(df["embedding"].to_list()).astype(np.float64)


def ks_shift_between(a_path: str | Path, b_path: str | Path, label: str = "") -> dict:
    """Per-dimension two-sample KS between embedding sets A and B."""
    A = _embeddings(a_path)
    B = _embeddings(b_path)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dim mismatch: {A.shape[1]} vs {B.shape[1]}")
    d = A.shape[1]

    ks_stats = np.empty(d)
    ks_pvals = np.empty(d)
    for j in range(d):
        ks_stats[j], ks_pvals[j] = ks_2samp(A[:, j], B[:, j])

    bonferroni = 0.01 / d
    n_sig = int((ks_pvals < bonferroni).sum())
    top = np.argsort(-ks_stats)[:50]
    result = {
        "label": label,
        "n_a": int(len(A)), "n_b": int(len(B)), "n_dims": d,
        "n_significant_ks": n_sig,
        "frac_significant": float(n_sig / d),
        "median_ks": float(np.median(ks_stats)),
        "mean_ks": float(np.mean(ks_stats)),
        "top_shifted_dims": [
            {"dim": int(j), "ks": float(ks_stats[j]), "pval": float(ks_pvals[j])} for j in top
        ],
    }
    logger.info("KS[%s]: %d/%d dims shifted (%.1f%%), median KS=%.4f",
                label or "?", n_sig, d, 100 * n_sig / d, result["median_ks"])
    return result


def run_ksshift(frozen_path: str | Path, adapted_path: str | Path,
                output_path: str | Path, label: str = "frozen-vs-adapted") -> dict:
    result = ks_shift_between(frozen_path, adapted_path, label=label)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
