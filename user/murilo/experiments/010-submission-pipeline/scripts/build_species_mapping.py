#!/usr/bin/env python3
"""Build species mapping between model outputs and BirdCLEF 2026 submission columns.

Computes overlap between model output species (from training data) and
234 BirdCLEF 2026 species (submission columns). Saves mapping JSON for use in
both ONNX export and Kaggle inference.

Works for both 2025-trained and 2026-trained models:
  - 2025-trained: 41/206 species overlap with 2026 submission columns
  - 2026-trained: 206/206 species overlap (all focal species map directly)

Usage:
    # For 2025-trained models (41 overlap)
    python scripts/build_species_mapping.py \
        --train-csv /path/to/birdclef-2025/train.csv \
        --output artifacts/weights/species_mapping.json

    # For 2026-trained models (206 overlap)
    python scripts/build_species_mapping.py \
        --train-csv /path/to/birdclef-2026/train.csv \
        --output artifacts/weights/species_mapping_2026.json
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_species_list(metadata_csv: str | Path) -> list[str]:
    """Extract sorted list of unique species from BirdCLEF 2025 metadata CSV."""
    df = pd.read_csv(metadata_csv)
    species = sorted(df["primary_label"].unique().tolist())
    return species


def main():
    parser = argparse.ArgumentParser(description="Build model→2026 species mapping")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2025/train.csv",
        help="Train CSV for the model's training data (2025 or 2026)",
    )
    parser.add_argument(
        "--submission-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/sample_submission.csv",
        help="BirdCLEF 2026 sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/weights/species_mapping.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Model species: sorted species list (matches model output dimension order)
    model_species = build_species_list(args.train_csv)
    logger.info(f"Model species: {len(model_species)} (from {args.train_csv})")

    # BirdCLEF 2026: species from submission columns (excluding row_id)
    sub_df = pd.read_csv(args.submission_csv, nrows=1)
    species_2026 = [c for c in sub_df.columns if c != "row_id"]
    logger.info(f"BirdCLEF 2026: {len(species_2026)} species")

    # Compute overlap
    set_model = set(model_species)
    set_2026 = set(species_2026)
    overlap = set_model & set_2026
    logger.info(f"Overlap: {len(overlap)} species")
    logger.info(f"  Model-only: {len(set_model - set_2026)} species")
    logger.info(f"  2026-only: {len(set_2026 - set_model)} species")

    # Build mapping: model output index → 2026 column name
    # Only for species that exist in both the model output and 2026 submission
    idx_to_name_2026 = {}
    for idx, sp in enumerate(model_species):
        if sp in set_2026:
            idx_to_name_2026[str(idx)] = sp

    logger.info(f"Mapped {len(idx_to_name_2026)} model output indices to 2026 species")

    # Save mapping (idx_to_name_2026 is the canonical key;
    # idx_2025_to_name_2026 kept for backward compatibility with existing Kaggle kernels)
    mapping = {
        "model_species": model_species,
        "birdclef2026_species": species_2026,
        "idx_to_name_2026": idx_to_name_2026,
        "idx_2025_to_name_2026": idx_to_name_2026,  # backward compat
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Species mapping saved to {output_path}")

    # Print overlap species for verification
    logger.info("Overlapping species:")
    for sp in sorted(overlap):
        idx = model_species.index(sp)
        logger.info(f"  [{idx:3d}] {sp}")


if __name__ == "__main__":
    main()
