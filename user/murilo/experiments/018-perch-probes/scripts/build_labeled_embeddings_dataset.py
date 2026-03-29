#!/usr/bin/env python3
"""Build the birdclef-2026-perch-labeled-embeddings Kaggle dataset.

Converts PACE-extracted labeled soundscape data to the format used by
reference notebooks (jaejohn/perch-meta pattern):
- Raw logits (NOT sigmoid) for mapped species
- Raw embeddings (1536-dim)
- Metadata with site/hour

Source (PACE):
  shared/embeddings/birdclef-2026/perch-v2/labeled/perch_embeddings.parquet   (792, 1540)
  shared/embeddings/birdclef-2026/perch-v2/labeled/perch_logits.parquet       (792, 14799)

Output (for Kaggle upload):
  labeled_perch_arrays.npz    — scores_raw (792, 234) RAW LOGITS + emb (792, 1536)
  labeled_perch_meta.parquet  — filename, window_idx, window_start, window_end, site, hour

IMPORTANT: scores_raw contains RAW LOGITS, not sigmoid probabilities.
The scoring notebook applies sigmoid only at the final submission step.

Usage:
  cd user/murilo/experiments/018-perch-probes
  python scripts/build_labeled_embeddings_dataset.py
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

# Species mapping (for logit → 234 BirdCLEF mapping)
SPECIES_MAPPING_PATH = Path(
    "/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026"
    "/user/murilo/experiments/018-perch-probes/artifacts/perch_species_mapping.json"
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


def map_logits_to_species(raw_logits: np.ndarray, species_mapping: dict) -> np.ndarray:
    """Map 14,795-dim Perch logits to 234 BirdCLEF species as RAW LOGITS (no sigmoid).

    Same mapping as compute_perch_mapped_scores() but WITHOUT sigmoid.
    Perch v2 outputs 14,795 logits. labels.csv has 14,796 rows (row 0 = non-bird).
    Model output index = labels.csv index - 1.
    """
    birdclef_species = species_mapping["birdclef_species"]
    direct_map = species_mapping["direct_map"]
    genus_proxy_map = species_mapping["genus_proxy_map"]
    n_species = len(birdclef_species)
    n_samples = raw_logits.shape[0]
    n_logits = raw_logits.shape[1]

    # Output: raw logits for 234 species (default -10 for unmapped = very low probability)
    scores = np.full((n_samples, n_species), -10.0, dtype=np.float32)

    for i, label in enumerate(birdclef_species):
        if label in direct_map:
            model_idx = direct_map[label] - 1
            if 0 <= model_idx < n_logits:
                scores[:, i] = raw_logits[:, model_idx].astype(np.float32)
        elif label in genus_proxy_map:
            valid = [idx - 1 for idx in genus_proxy_map[label] if 0 <= idx - 1 < n_logits]
            if valid:
                # Max of raw logits across genus relatives
                scores[:, i] = raw_logits[:, valid].max(axis=1).astype(np.float32)
        # else: unmapped → stays at -10.0 (very low logit)

    return scores


def main():
    print(f"Source: {EMBEDDINGS_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Load PACE parquets
    emb_df = pd.read_parquet(EMBEDDINGS_DIR / "perch_embeddings.parquet")
    logits_df = pd.read_parquet(EMBEDDINGS_DIR / "perch_logits.parquet")

    print(f"\nEmbeddings: {emb_df.shape}")
    print(f"Raw logits: {logits_df.shape}")

    # Load species mapping
    with open(SPECIES_MAPPING_PATH) as f:
        species_mapping = json.load(f)
    birdclef_species = species_mapping["birdclef_species"]
    print(f"Species: {len(birdclef_species)}")

    # Extract embedding numpy array
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    embeddings = emb_df[emb_cols].values.astype(np.float32)
    print(f"Embeddings array: {embeddings.shape}")

    # Extract raw logits and map to 234 species (NO sigmoid)
    logit_cols = [c for c in logits_df.columns if c.startswith("logit_")]
    raw_logits = logits_df[logit_cols].values.astype(np.float32)
    print(f"Raw logits array: {raw_logits.shape}")

    scores_raw = map_logits_to_species(raw_logits, species_mapping)
    print(f"Mapped scores (raw logits): {scores_raw.shape}")
    print(
        f"  Range: [{scores_raw.min():.2f}, {scores_raw.max():.2f}] (should be unbounded, not [0,1])"
    )

    # Build metadata with site/hour
    meta = emb_df[["filename", "window_idx", "window_start", "window_end"]].copy()
    sites_hours = meta["filename"].apply(lambda fn: pd.Series(parse_site_hour(fn)))
    meta["site"] = sites_hours[0]
    meta["hour"] = sites_hours[1].astype("Int64")

    print(f"Metadata: {meta.shape}")
    print(f"  Unique files: {meta['filename'].nunique()}")
    print(f"  Unique sites: {meta['site'].nunique()}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = OUTPUT_DIR / "labeled_perch_arrays.npz"
    np.savez_compressed(npz_path, scores_raw=scores_raw, emb=embeddings)
    print(f"\nSaved {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")

    meta_path = OUTPUT_DIR / "labeled_perch_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    print(f"Saved {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")

    # Also save species order for verification
    species_path = OUTPUT_DIR / "species_order.json"
    with open(species_path, "w") as f:
        json.dump(birdclef_species, f)
    print(f"Saved {species_path}")

    # Dataset metadata for Kaggle
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
    print("  # or: kaggle datasets version -p . -m 'Raw logits (not sigmoid)' -r zip")


if __name__ == "__main__":
    main()
