#!/usr/bin/env python3
"""Certainty scalars — enrich triage parquet with per-sample prediction-spread metrics.

Adds four columns derived from the clipwise logits, each a different lens on
"how decisive is the model on this sample":

- `shannon_norm`    : Shannon entropy of sigmoid probs, renormalized to sum to 1.
                      Strong negative correlation with top1_prob (~-0.86), so it
                      is largely redundant with "how peaked is the top prediction".
                      Useful mostly for its low AUC on confident_wrong (peaked-
                      but-wrong samples look similar to correct_uncertain).
- `bernoulli_sum`   : Σ [-p log p - (1-p) log(1-p)] over all classes. Sums the
                      independent Bernoulli uncertainties; captures "how many
                      species are genuinely ambiguous", not just "how peaked is
                      the top-1". Ranks confident_wrong near mixed_uncertainty
                      (both have multiple classes firing weakly).
- `top1_top2_gap`   : logit margin between top-1 and top-2 classes. Big gap =
                      decisive. Matches loss as a failure predictor (AUC ~0.88).
- `n_active`        : count of species with prob > 0.1. Integer-valued proxy for
                      "how many species the model thinks are active". >1 for
                      multi-species recordings (cooccurrence_label_ambiguity
                      averages 3.76).

Writes columns back to the triage parquet in place (or to `--output` if given).
Kept separate from triage.py because these are output-distribution statistics;
triage.py enriches with LABEL metadata (secondary_labels, known labels).
Different kind of enrichment, different iteration cycle.

Usage:
    python certainty.py \\
        --triage data/triage_v5j.parquet \\
        --logits data/losses_v5j_logits.npy

    # Summary of the four scalars by triage_category:
    python certainty.py --triage data/triage_v5j.parquet --logits ... --summary
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_certainty_scalars(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-sample certainty scalars from clipwise logits.

    Args:
        logits: (N, num_classes) clipwise logits.

    Returns:
        Dict mapping column name to (N,) array.
    """
    probs = expit(logits)

    # Shannon entropy of renormalized probabilities.
    # Multi-label sigmoid probs don't sum to 1, so this is a proxy — useful but
    # largely redundant with top1_prob (corr ~-0.86 on v5j val).
    probs_norm = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
    shannon_norm = -(probs_norm * np.log(probs_norm + 1e-12)).sum(axis=1)

    # Sum of Bernoulli entropies — more principled for multi-label output.
    # Captures "how many classes are genuinely uncertain" across all 234.
    p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
    bernoulli_sum = -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip)).sum(axis=1)

    # Logit margin between top-1 and top-2.
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    top1_top2_gap = sorted_logits[:, 0] - sorted_logits[:, 1]

    # Active-class count.
    n_active = (probs > 0.1).sum(axis=1)

    return {
        "shannon_norm": shannon_norm.astype(np.float32),
        "bernoulli_sum": bernoulli_sum.astype(np.float32),
        "top1_top2_gap": top1_top2_gap.astype(np.float32),
        "n_active": n_active.astype(np.int32),
    }


def enrich_triage(triage_path: Path, logits_path: Path, output_path: Path) -> pd.DataFrame:
    logger.info(f"Loading {triage_path}")
    df = pd.read_parquet(triage_path)
    logger.info(f"Loading {logits_path}")
    logits = np.load(logits_path)
    assert (
        len(df) == logits.shape[0]
    ), f"Row count mismatch: triage={len(df)}, logits={logits.shape[0]}"

    scalars = compute_certainty_scalars(logits)
    for col, arr in scalars.items():
        df[col] = arr

    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df)} rows with {len(scalars)} new columns to {output_path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Certainty scalars by triage category (sorted by mean_loss) ===")
    agg = (
        df.groupby("triage_category")
        .agg(
            n=("loss", "size"),
            mean_loss=("loss", "mean"),
            mean_top1_prob=("top1_prob", "mean"),
            mean_shannon_norm=("shannon_norm", "mean"),
            mean_bernoulli_sum=("bernoulli_sum", "mean"),
            mean_top1_top2_gap=("top1_top2_gap", "mean"),
            mean_n_active=("n_active", "mean"),
        )
        .sort_values("mean_loss")
    )
    print(agg.round(3).to_string())

    # AUC as predictor of 'real failure' (sanity check the scalars are meaningful)
    try:
        from sklearn.metrics import roc_auc_score

        y = df["triage_category"].isin(["mixed_uncertainty", "confident_wrong", "gt_invisible"])
        print("\n=== AUC as 'real failure' predictor ===")
        # For top1_prob: higher = less likely to be failure, so flip sign for AUC.
        for col, flip in [
            ("loss", False),
            ("shannon_norm", False),
            ("bernoulli_sum", False),
            ("top1_top2_gap", True),
            ("top1_prob", True),
        ]:
            score = -df[col] if flip else df[col]
            auc = roc_auc_score(y, score)
            print(f"  {col:16s} AUC={auc:.4f}")
    except ImportError:
        print("(scikit-learn not installed, skipping AUC)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--triage", type=Path, required=True, help="Triage parquet (from triage.py)")
    p.add_argument(
        "--logits",
        type=Path,
        default=None,
        help="Logits .npy. Defaults to losses_<model>_logits.npy next to the triage parquet.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet. Defaults to overwriting --triage in place.",
    )
    p.add_argument("--summary", action="store_true", help="Print category summary after enriching.")
    args = p.parse_args()

    if args.logits is None:
        # Infer from sibling filename: triage_<model>.parquet → losses_<model>_logits.npy
        stem = args.triage.stem.replace("triage_", "losses_", 1)
        args.logits = args.triage.parent / f"{stem}_logits.npy"
    if args.output is None:
        args.output = args.triage

    df = enrich_triage(args.triage, args.logits, args.output)
    if args.summary:
        print_summary(df)


if __name__ == "__main__":
    main()
