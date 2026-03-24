"""Spot check activity scores on a sample of focal recordings.

Picks 12 recordings (mix of species, durations, taxa) and prints per-chunk
activity scores alongside raw indices so we can verify the per-clip
normalization makes sense.

Usage:
    python spot_check_activity.py --activity artifacts/focal_activity.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data/birdclef-2026"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.activity)
    print(f"Loaded {len(df)} chunks from {df['filename'].nunique()} files")

    # Load metadata for taxa info
    meta = pd.read_csv(DATA_DIR / "train.csv")
    meta["basename"] = meta["filename"].apply(lambda f: Path(f).name)
    taxa_map = dict(zip(meta["basename"], meta["class_name"]))
    duration_map = {}
    for fname, group in df.groupby("filename"):
        duration_map[fname] = group["chunk_end"].max()

    # Pick 12 diverse clips:
    # - 2 short (<10s = 1-2 chunks)
    # - 2 medium (15-30s = 3-6 chunks)
    # - 2 long (>60s = 12+ chunks)
    # - 2 with high variance in activity (interesting structure)
    # - 2 with low variance (monotone)
    # - 2 random

    stats = df.groupby("filename").agg(
        n_chunks=("activity_score", "count"),
        mean_activity=("activity_score", "mean"),
        std_activity=("activity_score", "std"),
        min_activity=("activity_score", "min"),
        max_activity=("activity_score", "max"),
        max_time=("chunk_end", "max"),
    ).reset_index()
    stats["std_activity"] = stats["std_activity"].fillna(0)

    samples = []
    rng = np.random.RandomState(42)

    # Short clips (1-2 chunks)
    short = stats[stats["n_chunks"] <= 2]
    if len(short) >= 2:
        samples.extend(short.sample(2, random_state=rng)["filename"].tolist())

    # Medium clips (3-6 chunks)
    medium = stats[(stats["n_chunks"] >= 3) & (stats["n_chunks"] <= 6)]
    if len(medium) >= 2:
        samples.extend(medium.sample(2, random_state=rng)["filename"].tolist())

    # Long clips (12+ chunks)
    long = stats[stats["n_chunks"] >= 12]
    if len(long) >= 2:
        samples.extend(long.sample(2, random_state=rng)["filename"].tolist())

    # High variance (interesting structure)
    high_var = stats[stats["n_chunks"] >= 4].nlargest(20, "std_activity")
    if len(high_var) >= 2:
        samples.extend(high_var.sample(2, random_state=rng)["filename"].tolist())

    # Low variance (monotone)
    low_var = stats[(stats["n_chunks"] >= 4) & (stats["std_activity"] < 0.05)]
    if len(low_var) >= 2:
        samples.extend(low_var.sample(2, random_state=rng)["filename"].tolist())

    # Random
    remaining = stats[~stats["filename"].isin(samples)]
    if len(remaining) >= 2:
        samples.extend(remaining.sample(2, random_state=rng)["filename"].tolist())

    # Deduplicate
    samples = list(dict.fromkeys(samples))[:12]

    # Print detailed view for each
    for i, fname in enumerate(samples):
        clip = df[df["filename"] == fname].sort_values("chunk_start")
        species = clip["species"].iloc[0] if "species" in clip.columns and pd.notna(clip["species"].iloc[0]) else "?"
        taxa = taxa_map.get(fname, "?")
        dur = clip["chunk_end"].max()
        n_chunks = len(clip)

        clip_stats = stats[stats["filename"] == fname].iloc[0]

        print(f"\n{'='*80}")
        print(f"[{i+1}/12] {fname}")
        print(f"  Species: {species}  Taxa: {taxa}  Duration: {dur:.0f}s  Chunks: {n_chunks}")
        print(f"  Activity: mean={clip_stats['mean_activity']:.3f} std={clip_stats['std_activity']:.3f} "
              f"range=[{clip_stats['min_activity']:.3f}, {clip_stats['max_activity']:.3f}]")
        print()
        print(f"  {'Start':>6} {'End':>6} {'Activity':>9} {'SC_mean':>8} {'SC_max':>8} {'BioIdx':>8} {'NDSI':>8} {'Flat':>8}")
        print(f"  {'-'*65}")

        for _, row in clip.iterrows():
            # Visual bar for activity
            bar_len = int(row["activity_score"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(
                f"  {row['chunk_start']:>5.0f}s {row['chunk_end']:>5.0f}s "
                f"{row['activity_score']:>8.3f}  "
                f"{row['spectral_contrast_mean']:>8.4f} {row['spectral_contrast_max']:>8.4f} "
                f"{row['bioacoustic_index_mean']:>8.1f} {row['ndsi_mean']:>8.3f} "
                f"{row['spectral_flatness_mean']:>8.4f}  "
                f"{bar}"
            )

    # Overall summary
    print(f"\n{'='*80}")
    print(f"Overall: {len(df)} chunks, {df['filename'].nunique()} files")
    print(f"  Activity distribution:")
    for pct in [0, 10, 25, 50, 75, 90, 100]:
        val = df["activity_score"].quantile(pct / 100)
        print(f"    p{pct:>3}: {val:.3f}")

    # How many single-chunk files (can't normalize)?
    single_chunk = stats[stats["n_chunks"] == 1]
    print(f"\n  Single-chunk files: {len(single_chunk)} ({len(single_chunk)/len(stats)*100:.1f}%)")
    print(f"  These all get activity_score=0.5 (no within-clip variation)")


if __name__ == "__main__":
    main()
