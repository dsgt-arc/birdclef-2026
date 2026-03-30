#!/usr/bin/env python3
"""Merge pseudo-label shard parquets into a single file.

Usage:
    python scripts/merge_pseudo_labels.py \
        --parts-dir artifacts/pseudo_labels/parts/ \
        --output artifacts/pseudo_labels/soundscape_clip_probs.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", default="artifacts/pseudo_labels/parts/")
    parser.add_argument("--output", default="artifacts/pseudo_labels/soundscape_clip_probs.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parts = sorted(Path(args.parts_dir).glob("shard_*.parquet"))
    logger.info(f"Found {len(parts)} shard files")

    if not parts:
        logger.error("No shard files found!")
        return

    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged: {len(df)} rows from {len(parts)} shards")

    # Deduplicate just in case
    before = len(df)
    df = df.drop_duplicates(subset=["filename", "chunk_start"])
    if len(df) < before:
        logger.warning(f"Removed {before - len(df)} duplicate rows")

    # Sort
    df = df.sort_values(["filename", "chunk_start"]).reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Summary
    n_files = df["filename"].nunique()
    print(f"\n=== Pseudo-Label Summary ===")
    print(f"Total windows: {len(df)}")
    print(f"Total files: {n_files}")
    print(f"Windows per file: {len(df) / max(n_files, 1):.1f}")
    print(f"\nmax_prob distribution:")
    print(f"  mean={df['max_prob'].mean():.3f}, median={df['max_prob'].median():.3f}")
    print(f"  p10={df['max_prob'].quantile(0.1):.3f}, p90={df['max_prob'].quantile(0.9):.3f}")
    print(f"\nSpecies above threshold:")
    print(f"  >0.1: mean={df['n_species_above_01'].mean():.1f} per window")
    print(f"  >0.2: mean={df['n_species_above_02'].mean():.1f} per window")
    print(f"\nTop species frequency:")
    print(df["top_species"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
