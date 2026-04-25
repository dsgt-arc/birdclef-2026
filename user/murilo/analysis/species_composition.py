#!/usr/bin/env python3
"""Species composition — where are failures concentrated?

Takes an enriched triage parquet and asks: for each `triage_category`, which
species dominate? Specifically:

1. `--summary` (default): per-category top-N species table.
2. `--compare A B`: side-by-side comparison of two categories by species,
   sorted by over/under-representation. Answers "are mixed_uncertainty samples
   concentrated in species that correct_uncertain samples are also in, or are
   they a different set of species?"

The second mode is the one that informs fixes:
- If the same species dominate both categories → the model is under-confident
  generally, not species-specific. Ensemble/calibration is the lever.
- If different species dominate → failures are concentrated. Rare-species
  rebalancing or hard-negative mining on specific pairs is the lever.

Output to stdout (table) and optionally to CSV for downstream analysis.

Usage:
    # Per-category top-20 species:
    python species_composition.py --triage data/triage_v5j.parquet --top-n 20

    # Which species over-represent in mixed_uncertainty relative to correct_uncertain:
    python species_composition.py --triage data/triage_v5j.parquet \\
        --compare mixed_uncertainty correct_uncertain --top-n 20

    # Save the comparison to CSV:
    python species_composition.py --triage data/triage_v5j.parquet \\
        --compare mixed_uncertainty correct_uncertain \\
        --output data/species_composition_mixed_vs_correct.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def per_category_top_species(df: pd.DataFrame, top_n: int) -> None:
    print("\n=== Species distribution per triage_category (top-N by count) ===")
    for cat in df["triage_category"].unique():
        sub = df[df["triage_category"] == cat]
        counts = sub["primary_label"].value_counts().head(top_n)
        share = counts / len(sub)
        table = pd.DataFrame({"n": counts, "share": share.round(3)})
        print(f"\n--- {cat}  (n={len(sub)}, {df['primary_label'].nunique()} species in cat) ---")
        print(table.to_string())


def compare_categories(
    df: pd.DataFrame,
    cat_a: str,
    cat_b: str,
    top_n: int,
    output: Path | None,
) -> pd.DataFrame:
    """Compare species composition of two categories.

    For each species, compute:
      - n_a, n_b: count in each category
      - share_a, share_b: share of each category the species represents
      - log2_ratio: log2(share_a / share_b) — positive = over-represented in A
      - total: n_a + n_b (for sorting / filtering noise)
    """
    a = df[df["triage_category"] == cat_a]
    b = df[df["triage_category"] == cat_b]
    if len(a) == 0 or len(b) == 0:
        raise SystemExit(f"Category empty: {cat_a} n={len(a)}, {cat_b} n={len(b)}")

    counts_a = a["primary_label"].value_counts()
    counts_b = b["primary_label"].value_counts()

    all_species = sorted(set(counts_a.index) | set(counts_b.index))
    rows = []
    for sp in all_species:
        na, nb = int(counts_a.get(sp, 0)), int(counts_b.get(sp, 0))
        # Small-count smoothing to avoid log2(0): add 0.5 to both.
        sa = (na + 0.5) / (len(a) + 0.5)
        sb = (nb + 0.5) / (len(b) + 0.5)
        import math

        log2r = math.log2(sa / sb)
        rows.append(
            {
                "species": sp,
                f"n_{cat_a}": na,
                f"n_{cat_b}": nb,
                f"share_{cat_a}": round(sa, 4),
                f"share_{cat_b}": round(sb, 4),
                "log2_ratio_a_over_b": round(log2r, 3),
                "total": na + nb,
            }
        )
    out = pd.DataFrame(rows)

    # Species that over-represent in A (most problematic for A)
    over_in_a = out[out["total"] >= 5].sort_values("log2_ratio_a_over_b", ascending=False)
    # Species that over-represent in B
    over_in_b = out[out["total"] >= 5].sort_values("log2_ratio_a_over_b", ascending=True)

    print(
        f"\n=== {cat_a} (n={len(a)}, {counts_a.shape[0]} species) "
        f"vs {cat_b} (n={len(b)}, {counts_b.shape[0]} species) ==="
    )
    print(f"Total distinct species across both: {out.shape[0]}")
    print(f"Species with total >= 5 samples: {(out['total'] >= 5).sum()}")

    print(f"\n--- Top-{top_n} species OVER-represented in {cat_a} (most 'problem' species) ---")
    print(over_in_a.head(top_n).to_string(index=False))

    print(f"\n--- Top-{top_n} species OVER-represented in {cat_b} ---")
    print(over_in_b.head(top_n).to_string(index=False))

    # Concentration stats: what fraction of cat_a's samples come from the top-K species?
    top_k_share = {}
    for k in [10, 20, 50, 100]:
        if k <= counts_a.shape[0]:
            top_k_share[k] = counts_a.head(k).sum() / len(a)
    print(f"\n--- Concentration of {cat_a}: share of samples from top-K species ---")
    for k, s in top_k_share.items():
        print(f"  top-{k:3d} species: {s:.1%}")
    print(
        f"  (uniform-expected for {counts_a.shape[0]} species: "
        f"{10/counts_a.shape[0]:.1%} for top-10)"
    )

    if output:
        out.sort_values("log2_ratio_a_over_b", ascending=False).to_csv(output, index=False)
        logger.info(f"Wrote full comparison table ({len(out)} species) to {output}")

    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--triage", type=Path, required=True, help="Triage parquet")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("CAT_A", "CAT_B"),
        default=None,
        help="Compare species composition of two triage categories",
    )
    p.add_argument("--top-n", type=int, default=20, help="Rows to display")
    p.add_argument("--output", type=Path, default=None, help="CSV output path for --compare mode")
    args = p.parse_args()

    df = pd.read_parquet(args.triage)
    logger.info(f"Loaded {len(df)} samples from {args.triage}")

    if args.compare:
        compare_categories(df, args.compare[0], args.compare[1], args.top_n, args.output)
    else:
        per_category_top_species(df, args.top_n)


if __name__ == "__main__":
    main()
