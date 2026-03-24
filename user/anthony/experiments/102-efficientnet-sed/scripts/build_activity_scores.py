"""Build per-chunk activity scores for focal recordings.

For each potential 5s training chunk in BirdCLEF 2026 focal data, compute an
activity score from pre-computed acoustic indices (exp 010-2026). This score
represents confidence that the chunk contains an actual vocalization of the
labeled species (vs silence, noise, or background).

Output: parquet with columns [filename, chunk_start, chunk_end, activity_score,
spectral_contrast, bioacoustic_index, ndsi, spectral_flatness]

Usage:
    python build_activity_scores.py --output artifacts/focal_activity.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data/birdclef-2026"
INDICES_ROOT = Path("~/scratch/birdclef-2026/outputs/exp010-2026/labels/indices").expanduser()

CHUNK_DURATION = 5.0  # seconds
INDEX_FEATURES = ["ndsi", "spectral_contrast", "spectral_flatness", "bioacoustic_index"]


def load_indices(split: str = "train_audio") -> pd.DataFrame:
    """Load all acoustic index parquets for a split."""
    parts_dir = INDICES_ROOT / split / "parts"
    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquets in {parts_dir}")

    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df["filename"] = df["file"].apply(lambda f: Path(f).name)
    return df


def load_audio_durations() -> dict[str, float]:
    """Get duration of each focal recording from train.csv + audio_durations if available."""
    # Use the indices data itself to infer durations (max window_start + window_size)
    return None  # We'll compute from indices data


def aggregate_to_chunks(indices_df: pd.DataFrame, chunk_duration: float = 5.0) -> pd.DataFrame:
    """Aggregate per-window indices to chunk-level activity scores.

    Each chunk is a potential 5s training window. Indices are at 3s/1s resolution,
    so multiple index windows fall within each chunk.
    """
    # Compute chunk assignment for each index window
    indices_df = indices_df.copy()
    indices_df["chunk_start"] = (indices_df["window_start"] // chunk_duration) * chunk_duration

    # Aggregate: mean and max of each index within the chunk
    agg = indices_df.groupby(["filename", "chunk_start"]).agg(
        spectral_contrast_mean=("spectral_contrast", "mean"),
        spectral_contrast_max=("spectral_contrast", "max"),
        bioacoustic_index_mean=("bioacoustic_index", "mean"),
        bioacoustic_index_max=("bioacoustic_index", "max"),
        ndsi_mean=("ndsi", "mean"),
        spectral_flatness_mean=("spectral_flatness", "mean"),
        n_windows=("window_start", "count"),
    ).reset_index()

    agg["chunk_end"] = agg["chunk_start"] + chunk_duration

    return agg


def compute_activity_score(chunks: pd.DataFrame) -> pd.DataFrame:
    """Compute per-clip activity score from spectral contrast.

    Uses spectral_contrast_mean only — our best single predictor for call/no-call
    (0.888 AUROC in exp 101). Combined indices performed worse due to insufficient
    labeled data to learn proper weights.

    Normalized **per clip**: which frames in this recording have the most activity?
    This handles varying recording quality, background noise, and loudness without
    any species-level or global assumptions.
    """
    chunks = chunks.copy()

    col = "spectral_contrast_mean"

    def _normalize_group(group):
        values = group[col]
        lo = values.min()
        hi = values.max()
        if hi > lo:
            return ((values - lo) / (hi - lo)).clip(0, 1)
        else:
            # Single chunk or constant — no variation to normalize
            return pd.Series(0.5, index=group.index)

    chunks["activity_score"] = chunks.groupby("filename", group_keys=False).apply(
        _normalize_group
    )

    # Also compute a global score for cross-clip comparison
    p5 = chunks[col].quantile(0.05)
    p95 = chunks[col].quantile(0.95)
    if p95 > p5:
        chunks["activity_score_global"] = ((chunks[col] - p5) / (p95 - p5)).clip(0, 1)
    else:
        chunks["activity_score_global"] = 0.5

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/focal_activity.parquet"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading acoustic indices for focal recordings...")
    indices = load_indices("train_audio")
    print(f"  {len(indices)} index windows from {indices['filename'].nunique()} files")

    print("Aggregating to 5s chunks...")
    chunks = aggregate_to_chunks(indices, CHUNK_DURATION)
    print(f"  {len(chunks)} chunks from {chunks['filename'].nunique()} files")

    print("Computing activity scores...")
    chunks = compute_activity_score(chunks)

    # Add species info from filename path (species/recording.ogg)
    # The 'file' column has the full path including species dir
    file_to_species = {}
    for _, row in indices[["file", "filename"]].drop_duplicates().iterrows():
        parts = Path(row["file"]).parts
        # train_audio/species_id/recording.ogg
        for i, p in enumerate(parts):
            if p == "train_audio" and i + 1 < len(parts):
                file_to_species[row["filename"]] = parts[i + 1]
                break
    chunks["species"] = chunks["filename"].map(file_to_species)

    # Summary stats
    print(f"\nActivity score distribution:")
    print(f"  mean:   {chunks['activity_score'].mean():.3f}")
    print(f"  median: {chunks['activity_score'].median():.3f}")
    print(f"  p10:    {chunks['activity_score'].quantile(0.10):.3f}")
    print(f"  p90:    {chunks['activity_score'].quantile(0.90):.3f}")

    # Per-species summary
    species_stats = chunks.groupby("species")["activity_score"].agg(["mean", "median", "count"])
    print(f"\nPer-species activity (bottom 10 by median):")
    bottom = species_stats.nsmallest(10, "median")
    for sp, row in bottom.iterrows():
        print(f"  {sp}: median={row['median']:.3f}, mean={row['mean']:.3f}, chunks={int(row['count'])}")

    print(f"\nPer-species activity (top 10 by median):")
    top = species_stats.nlargest(10, "median")
    for sp, row in top.iterrows():
        print(f"  {sp}: median={row['median']:.3f}, mean={row['mean']:.3f}, chunks={int(row['count'])}")

    # Fraction of low-activity chunks (potential silence)
    low_activity = (chunks["activity_score"] < 0.2).mean()
    high_activity = (chunks["activity_score"] > 0.8).mean()
    print(f"\n  Low activity (<0.2):  {low_activity:.1%} of chunks")
    print(f"  High activity (>0.8): {high_activity:.1%} of chunks")

    # Save
    output_cols = [
        "filename", "species", "chunk_start", "chunk_end",
        "activity_score", "activity_score_global",
        "spectral_contrast_mean", "spectral_contrast_max",
        "bioacoustic_index_mean", "bioacoustic_index_max",
        "ndsi_mean", "spectral_flatness_mean",
        "n_windows",
    ]
    chunks[output_cols].to_parquet(args.output, index=False)
    print(f"\nSaved {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
