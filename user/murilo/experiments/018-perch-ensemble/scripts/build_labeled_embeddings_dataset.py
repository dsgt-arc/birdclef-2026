#!/usr/bin/env python3
"""Build the birdclef-2026-perch-labeled-embeddings Kaggle dataset.

Converts PACE-extracted labeled soundscape embeddings from parquet format to the
compact NPZ + parquet format used by reference notebooks (jaejohn/perch-meta pattern).

Source (PACE):
  shared/embeddings/birdclef-2026/perch-v2/labeled/perch_embeddings.parquet   (792, 1540)
  shared/embeddings/birdclef-2026/perch-v2/labeled/perch_mapped_scores.parquet (792, 238)

Output (for Kaggle upload):
  labeled_perch_arrays.npz    — embeddings (792, 1536) + mapped_scores (792, 234)
  labeled_perch_meta.parquet  — filename, window_idx, window_start, window_end, site, hour

Usage:
  cd user/murilo/experiments/018-perch-ensemble
  python scripts/build_labeled_embeddings_dataset.py

  # Then upload to Kaggle:
  cd /path/to/output
  kaggle datasets create -p . -r zip
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_USER = "murilogustineli"
DATASET_SLUG = "birdclef-2026-perch-labeled-embeddings"

# PACE source paths
EMBEDDINGS_DIR = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/embeddings/birdclef-2026/perch-v2/labeled"
)

# Output directory
OUTPUT_DIR = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/datasets/birdclef-2026/perch-labeled-embeddings"
)


def parse_site_hour(filename: str) -> tuple[str | None, int | None]:
    """Extract site and hour from soundscape filename."""
    m = re.match(r"BC2026_Train_\d+_(S\d+)_\d{8}_(\d{6})\.ogg", filename)
    if m:
        return m.group(1), int(m.group(2)[:2])
    return None, None


def main():
    print(f"Source: {EMBEDDINGS_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Load PACE parquets
    emb_df = pd.read_parquet(EMBEDDINGS_DIR / "perch_embeddings.parquet")
    ms_df = pd.read_parquet(EMBEDDINGS_DIR / "perch_mapped_scores.parquet")

    print(f"\nEmbeddings: {emb_df.shape}")
    print(f"Mapped scores: {ms_df.shape}")

    # Extract numpy arrays
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    embeddings = emb_df[emb_cols].values.astype(np.float32)

    species_cols = [
        c
        for c in ms_df.columns
        if c not in ["filename", "window_idx", "window_start", "window_end"]
    ]
    mapped_scores = ms_df[species_cols].values.astype(np.float32)

    print(f"Embeddings array: {embeddings.shape} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    print(
        f"Mapped scores array: {mapped_scores.shape} ({mapped_scores.nbytes / 1024 / 1024:.1f} MB)"
    )

    # Build metadata with site/hour
    meta = emb_df[["filename", "window_idx", "window_start", "window_end"]].copy()
    sites_hours = meta["filename"].apply(lambda fn: pd.Series(parse_site_hour(fn)))
    meta["site"] = sites_hours[0]
    meta["hour"] = sites_hours[1].astype("Int64")

    print(f"Metadata: {meta.shape}")
    print(f"  Unique files: {meta['filename'].nunique()}")
    print(f"  Unique sites: {meta['site'].nunique()}")
    print(f"  Species columns: {len(species_cols)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = OUTPUT_DIR / "labeled_perch_arrays.npz"
    np.savez_compressed(npz_path, embeddings=embeddings, mapped_scores=mapped_scores)
    print(f"\nSaved {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")

    meta_path = OUTPUT_DIR / "labeled_perch_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    print(f"Saved {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")

    # Also save species column order for downstream verification
    species_path = OUTPUT_DIR / "species_order.json"
    with open(species_path, "w") as f:
        json.dump(species_cols, f)
    print(f"Saved {species_path}")

    # Create dataset-metadata.json for Kaggle
    dataset_meta = {
        "title": DATASET_SLUG,
        "id": f"{KAGGLE_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_json_path = OUTPUT_DIR / "dataset-metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"Saved {meta_json_path}")

    print("\n--- Ready for upload ---")
    print(f"  cd {OUTPUT_DIR}")
    print("  kaggle datasets create -p . -r zip")


if __name__ == "__main__":
    main()
