#!/usr/bin/env python3
"""D1 diagnostic: compare v5j (LSEHead) and v6k (AttHead) attention behavior
on highest-loss-during-training and lowest-loss-during-training examples.

Operates on outputs from `precompute_losses.py --data-split=train` for
both models. Buckets samples by training-loss percentile (top 1%, bottom 1%,
mid 50% as reference) and compares per-sample attention metrics across
buckets and across models.

The discriminator question (H1 information vs H2 modeling):

- If high-loss train samples show DIFFUSE attention on both models →
  pattern is data-driven (model honestly can't localize because the call
  is weak/absent). H1 strengthens.
- If v6k (AttHead) shows PEAKED-but-WRONG attention on high-loss samples
  where v5j (LSEHead) shows diffuse → AttHead exposes a head-level
  pathology that LSE-implicit attention masks. H2 candidate.
- If v6k attends to the right frame on low-loss samples (peaked, correct)
  but not on high-loss → head is *capable*, just under-pressured by the
  training objective. H2 supported with a specific testable hypothesis.

Usage:
    cd user/murilo/analysis
    python train_side_analysis.py \\
        --v5j-losses data/losses_v5j_train.parquet \\
        --v5j-perframe data/losses_v5j_train_perframe_logits.npy \\
        --v6k-losses data/losses_v6k_train.parquet \\
        --v6k-perframe data/losses_v6k_train_perframe_logits.npy \\
        --v6k-perframe-att data/losses_v6k_train_perframe_att.npy \\
        --output-dir data/d1_train_side/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from per_frame_analysis import compute_per_sample_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def add_loss_buckets(df: pd.DataFrame, top_pct: float = 1.0) -> pd.DataFrame:
    """Add a `loss_bucket` column: 'top_pct', 'bottom_pct', or 'mid'.

    top_pct=1.0 → top 1% by loss is bucket 'top_1pct', bottom 1% is 'bottom_1pct'.
    """
    df = df.copy()
    n = len(df)
    n_extreme = max(1, int(n * top_pct / 100))

    # Sort by loss and tag extremes
    sorted_idx = df["loss"].sort_values().index
    bucket = np.array(["mid"] * n, dtype=object)
    bucket[sorted_idx[:n_extreme]] = f"bottom_{int(top_pct)}pct"
    bucket[sorted_idx[-n_extreme:]] = f"top_{int(top_pct)}pct"
    df["loss_bucket"] = bucket
    return df


def summarize_by_bucket(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Group by loss_bucket and compute mean/median for each metric column."""
    rows = []
    for bucket, sub in df.groupby("loss_bucket"):
        row = {
            "loss_bucket": bucket,
            "n": len(sub),
            "mean_loss": sub["loss"].mean(),
            "median_loss": sub["loss"].median(),
        }
        for col in metric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_median"] = sub[col].median()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("loss_bucket")
    # Order: bottom, mid, top
    bucket_order = sorted(
        out.index, key=lambda x: (0 if x.startswith("bottom") else 2 if x.startswith("top") else 1)
    )
    return out.reindex(bucket_order).round(3)


def cross_tabulate(
    v5j_df: pd.DataFrame, v6k_df: pd.DataFrame, top_pct: float = 1.0
) -> pd.DataFrame:
    """Cross-tab: for each (v5j bucket, v6k bucket) pair, count samples and
    aggregate metrics.

    Specifically: how many samples that are in v5j's top 1% are ALSO in v6k's
    top 1% — i.e. is the failure agreement strong (data-driven failures) or
    weak (model-specific failures)?
    """
    if len(v5j_df) != len(v6k_df):
        raise ValueError(
            f"v5j ({len(v5j_df)}) and v6k ({len(v6k_df)}) row counts differ. "
            "Inputs must come from the same training set with same seed."
        )
    # Assume `idx` aligns across both
    merged = v5j_df[["idx", "loss", "loss_bucket"]].merge(
        v6k_df[["idx", "loss", "loss_bucket"]],
        on="idx",
        suffixes=("_v5j", "_v6k"),
    )
    crosstab = pd.crosstab(merged["loss_bucket_v5j"], merged["loss_bucket_v6k"], margins=True)
    return crosstab


def cross_compare_attention(
    v5j_df: pd.DataFrame,
    v6k_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each loss bucket, compare v5j's LSE-implicit attention metrics
    against v6k's learned attention metrics on the SAME samples.

    Aligns via `idx`. For each bucket, reports v5j metric, v6k metric, and
    delta. The delta is the head-level discriminator.
    """
    # Tag with bucket from v5j (so we're conditioning on training difficulty
    # as v5j sees it)
    merged = v5j_df.rename(
        columns={
            "gt_attn_entropy": "gt_attn_entropy_v5j",
            "gt_attn_max": "gt_attn_max_v5j",
            "gt_logit_std": "gt_logit_std_v5j",
            "top1_majority_frac": "top1_majority_frac_v5j",
            "loss": "loss_v5j",
        }
    ).merge(
        v6k_df[
            [
                "idx",
                "gt_attn_entropy",
                "gt_attn_max",
                "gt_logit_std",
                "top1_majority_frac",
                "loss",
            ]
        ].rename(
            columns={
                "gt_attn_entropy": "gt_attn_entropy_v6k",
                "gt_attn_max": "gt_attn_max_v6k",
                "gt_logit_std": "gt_logit_std_v6k",
                "top1_majority_frac": "top1_majority_frac_v6k",
                "loss": "loss_v6k",
            }
        ),
        on="idx",
    )

    rows = []
    for bucket, sub in merged.groupby("loss_bucket"):
        row = {"loss_bucket": bucket, "n": len(sub)}
        for metric in ("gt_attn_entropy", "gt_attn_max", "gt_logit_std", "top1_majority_frac"):
            v5j_val = sub[f"{metric}_v5j"].mean()
            v6k_val = sub[f"{metric}_v6k"].mean()
            row[f"{metric}_v5j"] = round(v5j_val, 3)
            row[f"{metric}_v6k"] = round(v6k_val, 3)
            row[f"{metric}_delta"] = round(v6k_val - v5j_val, 3)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("loss_bucket")
    bucket_order = sorted(
        out.index, key=lambda x: (0 if x.startswith("bottom") else 2 if x.startswith("top") else 1)
    )
    return out.reindex(bucket_order)


def main():
    parser = argparse.ArgumentParser(description="D1 train-side attention analysis (v5j vs v6k)")
    parser.add_argument("--v5j-losses", type=str, required=True)
    parser.add_argument("--v5j-perframe", type=str, required=True)
    parser.add_argument("--v6k-losses", type=str, required=True)
    parser.add_argument("--v6k-perframe", type=str, required=True)
    parser.add_argument("--v6k-perframe-att", type=str, required=True)
    parser.add_argument(
        "--species-list-source",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv",
    )
    parser.add_argument("--top-pct", type=float, default=1.0)
    parser.add_argument(
        "--lse-temperature", type=float, default=1.0, help="v5j lse_temperature (1.0)"
    )
    parser.add_argument("--output-dir", type=str, default="data/d1_train_side/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build species list
    taxonomy = pd.read_csv(args.species_list_source)
    species_list = sorted(taxonomy["primary_label"].astype(str).unique().tolist())
    logger.info(f"Species list: {len(species_list)} species")

    # ------------------------------------------------------------------
    # v5j (LSE-implicit attention)
    # ------------------------------------------------------------------
    logger.info("Loading v5j artifacts...")
    v5j_losses = pd.read_parquet(args.v5j_losses)
    v5j_perframe = np.load(args.v5j_perframe)
    logger.info(f"v5j: {len(v5j_losses)} samples, perframe shape={v5j_perframe.shape}")

    logger.info("Computing v5j per-sample metrics (LSE-implicit attention)...")
    v5j_enriched = compute_per_sample_metrics(
        v5j_losses,
        v5j_perframe,
        species_list,
        lse_temperature=args.lse_temperature,
        head_type="lse",
    )
    v5j_enriched = add_loss_buckets(v5j_enriched, top_pct=args.top_pct)
    v5j_enriched.to_parquet(output_dir / "v5j_train_enriched.parquet", index=False)

    # ------------------------------------------------------------------
    # v6k (AttHead learned attention)
    # ------------------------------------------------------------------
    logger.info("Loading v6k artifacts...")
    v6k_losses = pd.read_parquet(args.v6k_losses)
    v6k_perframe = np.load(args.v6k_perframe)
    v6k_perframe_att = np.load(args.v6k_perframe_att)
    logger.info(
        f"v6k: {len(v6k_losses)} samples, perframe shape={v6k_perframe.shape}, "
        f"perframe_att shape={v6k_perframe_att.shape}"
    )

    logger.info("Computing v6k per-sample metrics (learned softmax(tanh(att_raw)))...")
    v6k_enriched = compute_per_sample_metrics(
        v6k_losses,
        v6k_perframe,
        species_list,
        head_type="atthead",
        perframe_att=v6k_perframe_att,
    )
    v6k_enriched = add_loss_buckets(v6k_enriched, top_pct=args.top_pct)
    v6k_enriched.to_parquet(output_dir / "v6k_train_enriched.parquet", index=False)

    # ------------------------------------------------------------------
    # Per-bucket summaries
    # ------------------------------------------------------------------
    metric_cols = [
        "gt_attn_entropy",
        "gt_attn_max",
        "gt_logit_std",
        "top1_wander",
        "top1_majority_frac",
        "gt_prob",
    ]

    print("\n=== v5j (LSEHead) per-bucket summary ===")
    v5j_summary = summarize_by_bucket(v5j_enriched, metric_cols)
    print(v5j_summary.to_string())
    v5j_summary.to_csv(output_dir / "v5j_summary_by_bucket.csv")

    print("\n=== v6k (AttHead) per-bucket summary ===")
    v6k_summary = summarize_by_bucket(v6k_enriched, metric_cols)
    print(v6k_summary.to_string())
    v6k_summary.to_csv(output_dir / "v6k_summary_by_bucket.csv")

    # ------------------------------------------------------------------
    # Cross-tabulation: failure overlap between models
    # ------------------------------------------------------------------
    print("\n=== Loss-bucket cross-tabulation (v5j rows vs v6k cols) ===")
    crosstab = cross_tabulate(v5j_enriched, v6k_enriched, top_pct=args.top_pct)
    print(crosstab.to_string())
    crosstab.to_csv(output_dir / "loss_bucket_crosstab.csv")

    # ------------------------------------------------------------------
    # Same-sample attention comparison conditioned on v5j's bucket
    # ------------------------------------------------------------------
    print("\n=== Same-sample attention comparison (conditioned on v5j bucket) ===")
    cmp = cross_compare_attention(v5j_enriched, v6k_enriched)
    print(cmp.to_string())
    cmp.to_csv(output_dir / "same_sample_attention_comparison.csv")

    # ------------------------------------------------------------------
    # Reference values
    # ------------------------------------------------------------------
    T = v5j_perframe.shape[2]
    uniform_entropy = float(np.log(T))
    print(
        f"\nReference: T={T} frames; uniform-attention entropy = "
        f"log({T}) = {uniform_entropy:.3f} nats. "
        f"Perfectly peaked attention has entropy 0."
    )

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
