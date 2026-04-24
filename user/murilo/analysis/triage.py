#!/usr/bin/env python3
"""Tabular triage — enrich losses parquet with label metadata for failure-mode analysis.

For each sample in a losses_<model>.parquet, joins with train.csv to add
secondary_labels (for focal) or uses all_labels (for soundscape) and flags:

- `known_labels`     : primary + secondary species known to be present in the crop
- `top1_is_gt`       : model's top-1 == primary_label
- `top1_is_secondary`: top-1 is in the secondary labels (plausibly a real co-occurrence,
                       not a wrong prediction)
- `top1_is_known`    : top-1 is somewhere in the ground truth label set
- `saliency_informative`: gt_prob in [0.05, 0.9] — range where sigmoid isn't saturated
                          and gradient saliency is trustworthy
- `triage_category`  : heuristic bucket (see categorize())

This enables honest failure-mode analysis: "high loss" doesn't mean "model is wrong"
if the top-1 is a legitimate background species.

Usage:
    python triage.py --losses data/losses_v5j.parquet \\
                     --output data/triage_v5j.parquet

    # inspect a specific row in full detail
    python triage.py --losses data/losses_v5j.parquet --row-id 865
"""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CSV = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train.csv"
)


def parse_secondary_labels(val) -> list[str]:
    """Parse a secondary_labels cell into list[str]. Handles NaN, '[]', list literals."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    s = str(val).strip()
    if not s or s == "[]":
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def file_key_from_full_path(full_path: str) -> str:
    """Extract 'species_folder/basename.ogg' from a full filesystem path."""
    parts = Path(full_path).parts
    if len(parts) < 2:
        return full_path
    return f"{parts[-2]}/{parts[-1]}"


def build_triage_table(losses_path: Path, train_csv: Path) -> pd.DataFrame:
    logger.info(f"Loading {losses_path}")
    df = pd.read_parquet(losses_path)

    logger.info(f"Loading {train_csv}")
    train = pd.read_csv(train_csv)
    # Build lookup keyed by 'species_folder/basename.ogg' (the train.csv filename format).
    sec_lookup: dict[str, list[str]] = {
        row["filename"]: parse_secondary_labels(row["secondary_labels"])
        for _, row in train.iterrows()
    }

    df["file_key"] = df["filename"].apply(file_key_from_full_path)

    def get_known_labels(row: pd.Series) -> list[str]:
        if row["source"] == "focal":
            secs = sec_lookup.get(row["file_key"], [])
            # Primary first, unique, preserve order
            seen = set()
            out = []
            for x in [row["primary_label"], *secs]:
                if x and x != "unknown" and x not in seen:
                    out.append(x)
                    seen.add(x)
            return out
        # Soundscape: all_labels in the parquet is ';'-joined multi-hot
        labels = str(row.get("all_labels", "") or "").split(";")
        return [lab for lab in labels if lab and lab != "unknown"]

    df["known_labels"] = df.apply(get_known_labels, axis=1)
    df["n_known_labels"] = df["known_labels"].apply(len)

    df["top1_is_gt"] = df["top1_species"] == df["primary_label"]
    df["top1_is_secondary"] = df.apply(
        lambda r: (r["top1_species"] in r["known_labels"])
        and (r["top1_species"] != r["primary_label"]),
        axis=1,
    )
    df["top1_is_known"] = df.apply(lambda r: r["top1_species"] in r["known_labels"], axis=1)

    # Saliency reliability: sigmoid gradient is small when prob is near 0 or 1.
    df["saliency_informative"] = (df["gt_prob"] >= 0.05) & (df["gt_prob"] <= 0.9)

    df["triage_category"] = df.apply(categorize, axis=1)
    return df


def categorize(r: pd.Series) -> str:
    """Heuristic classification. Each category has a clear operational definition.

    - correct_confident: top-1 == GT AND gt_prob >= 0.9. Any loss comes from other
      classes, not from getting the primary wrong. Saliency is saturated → not useful.
    - correct_uncertain: top-1 == GT but gt_prob < 0.9. Correct-but-unsure; saliency
      is informative here (non-saturated) — best case for the GT-side analysis.
    - cooccurrence_label_ambiguity: top-1 is a secondary label of the recording.
      Model is probably picking up a real background species; not a misprediction
      in the usual sense. Loss just reflects the "didn't pick the primary" signal.
    - gt_invisible: gt_rank > 20 AND gt_prob < 0.05. Model has no signal for GT in
      this crop. Usually means the 5s crop doesn't contain the call, OR the species
      is undertrained, OR domain shift.
    - confident_wrong: top-1 NOT in any known label, top-1_prob >= 0.5, gt_rank >= 5.
      A genuine confident misidentification unrelated to the recording's labels.
    - mixed_uncertainty: everything else — low-confidence wrong predictions.
    """
    if r["top1_is_gt"]:
        return "correct_confident" if r["gt_prob"] >= 0.9 else "correct_uncertain"
    if r["top1_is_secondary"]:
        return "cooccurrence_label_ambiguity"
    if r["gt_rank"] > 20 and r["gt_prob"] < 0.05:
        return "gt_invisible"
    if r["top1_prob"] >= 0.5 and r["gt_rank"] >= 5:
        return "confident_wrong"
    return "mixed_uncertainty"


def print_summary(df: pd.DataFrame, top_n: int) -> None:
    print("\n" + "=" * 70)
    print("TRIAGE SUMMARY")
    print("=" * 70)
    print(
        f"Total: {len(df)}  (focal={int((df['source'] == 'focal').sum())}, "
        f"soundscape={int((df['source'] == 'soundscape').sum())})"
    )

    print("\n--- Category distribution (all samples) ---")
    agg = (
        df.groupby("triage_category")
        .agg(
            n=("loss", "size"),
            mean_loss=("loss", "mean"),
            median_loss=("loss", "median"),
        )
        .sort_values("n", ascending=False)
    )
    agg["pct"] = 100 * agg["n"] / len(df)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n--- Category distribution (top-{top_n} highest loss) ---")
    top = df.nlargest(top_n, "loss")
    top_counts = top["triage_category"].value_counts()
    for cat, n in top_counts.items():
        print(f"  {cat:32s} n={n:3d} ({100 * n / top_n:5.1f}%)")

    print("\n--- Saliency informativeness (all samples) ---")
    inf = df["saliency_informative"].sum()
    low = (df["gt_prob"] < 0.05).sum()
    high = (df["gt_prob"] > 0.9).sum()
    print(f"  prob in [0.05, 0.9] (informative): {inf:5d} ({100 * inf / len(df):5.1f}%)")
    print(f"  prob < 0.05 (saturated low):       {low:5d} ({100 * low / len(df):5.1f}%)")
    print(f"  prob > 0.9  (saturated high):      {high:5d} ({100 * high / len(df):5.1f}%)")

    print("\n--- Top-1 relation to labels (all samples) ---")
    print(
        f"  top-1 == GT:                  {df['top1_is_gt'].sum():5d} "
        f"({100 * df['top1_is_gt'].mean():.1f}%)"
    )
    print(f"  top-1 is secondary label:     {df['top1_is_secondary'].sum():5d}")
    print(f"  top-1 in any known label:     {df['top1_is_known'].sum():5d}")
    print(
        f"  top-1 NOT in any label:       {(~df['top1_is_known']).sum():5d} "
        f"({100 * (~df['top1_is_known']).mean():.1f}%)"
    )

    print(f"\n--- Top-{top_n} detail ---")
    cols = [
        "idx",
        "primary_label",
        "source",
        "loss",
        "gt_rank",
        "gt_prob",
        "top1_species",
        "top1_prob",
        "top1_is_secondary",
        "n_known_labels",
        "saliency_informative",
        "triage_category",
    ]
    with pd.option_context("display.max_colwidth", 30, "display.width", 200):
        print(top[cols].to_string(index=False))


def print_row_detail(df: pd.DataFrame, row_id: int) -> None:
    row = df[df["idx"] == row_id]
    if len(row) == 0:
        print(f"No row with idx={row_id}")
        return
    r = row.iloc[0]
    print("\n" + "=" * 70)
    print(f"ROW {row_id} DETAIL")
    print("=" * 70)
    print(f"file:                 {r['filename']}")
    print(f"source:               {r['source']}")
    print(f"primary_label (GT):   {r['primary_label']}")
    print(f"known_labels ({r['n_known_labels']}):   {r['known_labels']}")
    print(f"top1 species:         {r['top1_species']}  (prob={r['top1_prob']:.4f})")
    print(f"gt_prob:              {r['gt_prob']:.4f}  (rank={r['gt_rank']})")
    print(f"loss:                 {r['loss']:.4f}")
    print(f"offset_sec:           {r.get('offset_sec', 'n/a')}")
    print(f"top5:                 {r.get('top5_species', '')}")
    print(f"                      {r.get('top5_probs', '')}")
    print(f"top1_is_gt:           {r['top1_is_gt']}")
    print(f"top1_is_secondary:    {r['top1_is_secondary']}")
    print(f"saliency_informative: {r['saliency_informative']}")
    print(f"triage_category:      {r['triage_category']}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--losses", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: data/triage_<model>.parquet next to losses file",
    )
    p.add_argument("--top-n", type=int, default=20, help="Summary size")
    p.add_argument(
        "--row-id", type=int, default=None, help="If set, print detail for this row_id only"
    )
    args = p.parse_args()

    if args.output is None:
        stem = args.losses.stem.replace("losses_", "triage_", 1)
        args.output = args.losses.parent / f"{stem}.parquet"

    df = build_triage_table(args.losses, args.train_csv)
    df.to_parquet(args.output, index=False)
    logger.info(f"Wrote {len(df)} rows to {args.output}")

    if args.row_id is not None:
        print_row_detail(df, args.row_id)
    else:
        print_summary(df, args.top_n)


if __name__ == "__main__":
    main()
