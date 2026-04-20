"""Canonical 234-species ordering for BirdCLEF 2026 submissions.

Reads `perch_species_mapping.json` directly rather than importing from
`birdclef_106`, so the Kaggle kernel can ship this module without the
exp 106 dep chain (polars, sklearn, etc.). The canonical artifact lives
at `user/anthony/experiments/106-perch-bayesian/artifacts/perch_species_mapping.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parents[3]
    / "106-perch-bayesian"
    / "artifacts"
    / "perch_species_mapping.json"
)

NUM_SPECIES = 234


def load_species_mapping(mapping_path: str | Path | None = None) -> dict:
    """Read the perch_species_mapping.json artifact.

    Returns dict with keys including `birdclef_species` (234 labels),
    `direct_map`, `genus_proxy_map`, `unmapped`, `taxa_classes`.
    """
    if mapping_path is None:
        mapping_path = DEFAULT_MAPPING_PATH
    with open(mapping_path) as f:
        return json.load(f)


def build_species_list(mapping_path: str | Path | None = None) -> list[str]:
    """Return the sorted list of 234 BirdCLEF 2026 species labels."""
    mapping = load_species_mapping(mapping_path)
    species = sorted(mapping["birdclef_species"])
    if len(species) != NUM_SPECIES:
        raise ValueError(
            f"Expected {NUM_SPECIES} species, got {len(species)} from {mapping_path}"
        )
    return species
