#!/usr/bin/env python3
"""Build species mapping between BirdCLEF 2025 model outputs and 2026 submission columns.

Computes overlap between 206 BirdCLEF 2025 species (model output indices) and
234 BirdCLEF 2026 species (submission columns). Saves mapping JSON for use in
both ONNX export and Kaggle inference.

Usage:
    python scripts/build_species_mapping.py \
        --train-csv /path/to/birdclef-2025/train.csv \
        --submission-csv /path/to/birdclef-2026/sample_submission.csv \
        --output artifacts/weights/species_mapping.json
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
    parser = argparse.ArgumentParser(description="Build BirdCLEF 2025→2026 species mapping")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2025/train.csv",
        help="BirdCLEF 2025 train.csv",
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

    # BirdCLEF 2025: sorted species list (matches model output dimension order)
    species_2025 = build_species_list(args.train_csv)
    logger.info(f"BirdCLEF 2025: {len(species_2025)} species")

    # BirdCLEF 2026: species from submission columns (excluding row_id)
    sub_df = pd.read_csv(args.submission_csv, nrows=1)
    species_2026 = [c for c in sub_df.columns if c != "row_id"]
    logger.info(f"BirdCLEF 2026: {len(species_2026)} species")

    # Compute overlap
    set_2025 = set(species_2025)
    set_2026 = set(species_2026)
    overlap = set_2025 & set_2026
    logger.info(f"Overlap: {len(overlap)} species")
    logger.info(f"  2025-only: {len(set_2025 - set_2026)} species")
    logger.info(f"  2026-only: {len(set_2026 - set_2025)} species")

    # Build mapping: model output index (2025) → 2026 column name
    # Only for species that exist in both years
    idx_2025_to_name_2026 = {}
    for idx, sp in enumerate(species_2025):
        if sp in set_2026:
            idx_2025_to_name_2026[str(idx)] = sp

    logger.info(f"Mapped {len(idx_2025_to_name_2026)} model output indices to 2026 species")

    # Save mapping
    mapping = {
        "birdclef2025_species": species_2025,
        "birdclef2026_species": species_2026,
        "idx_2025_to_name_2026": idx_2025_to_name_2026,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Species mapping saved to {output_path}")

    # Print overlap species for verification
    logger.info("Overlapping species:")
    for sp in sorted(overlap):
        idx = species_2025.index(sp)
        logger.info(f"  [{idx:3d}] {sp}")


if __name__ == "__main__":
    main()
