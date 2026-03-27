"""Build Perch v2 -> BirdCLEF 2026 species mapping.

Reads:
  - BirdCLEF 2026 taxonomy.csv (234 species with scientific_name, class_name)
  - Perch v2 assets/labels.csv (14,795 scientific names, one per line)

Outputs:
  - artifacts/perch_species_mapping.json with:
    - direct_map: {birdclef_primary_label: perch_index}  (203 species)
    - genus_proxy_map: {birdclef_primary_label: [perch_idx1, perch_idx2, ...]}  (6 species)
    - unmapped: [primary_label, ...]  (25 insect sonotypes)
    - birdclef_species: ordered list of all 234 primary_labels
    - birdclef_to_scientific: {primary_label: scientific_name}
    - taxa_classes: {primary_label: class_name}  (Aves, Amphibia, Insecta, etc.)
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Perch v2 -> BirdCLEF 2026 species mapping")
    parser.add_argument(
        "--taxonomy-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv",
        help="Path to BirdCLEF 2026 taxonomy.csv",
    )
    parser.add_argument(
        "--perch-labels",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2/assets/labels.csv",
        help="Path to Perch v2 labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/ relative to experiment dir)",
    )
    return parser.parse_args()


def load_perch_labels(labels_path: str) -> dict[str, int]:
    """Load Perch v2 labels.csv -> {scientific_name: index}.

    labels.csv has one scientific name per line.
    Line 0 is 'inat2024_fsd50k' (non-bird class).
    """
    name_to_idx: dict[str, int] = {}
    with open(labels_path) as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                name_to_idx[name] = idx
    return name_to_idx


def build_genus_index(perch_labels: dict[str, int]) -> dict[str, list[int]]:
    """Build genus -> [perch_indices] lookup from Perch labels."""
    genus_to_indices: dict[str, list[int]] = {}
    for name, idx in perch_labels.items():
        # Skip non-species entries
        if idx == 0:
            continue
        parts = name.split()
        if len(parts) >= 2:
            genus = parts[0]
            genus_to_indices.setdefault(genus, []).append(idx)
    return genus_to_indices


def main() -> None:
    args = parse_args()

    # Resolve output directory
    if args.output_dir is None:
        experiment_dir = Path(__file__).resolve().parent.parent
        output_dir = experiment_dir / "artifacts"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BirdCLEF 2026 taxonomy
    taxonomy = pd.read_csv(args.taxonomy_csv)
    print(f"Loaded taxonomy: {len(taxonomy)} species")
    print(f"  Columns: {list(taxonomy.columns)}")
    print("  Class distribution:")
    for cls, count in taxonomy["class_name"].value_counts().items():
        print(f"    {cls}: {count}")

    # Load Perch v2 labels
    perch_labels = load_perch_labels(args.perch_labels)
    print(f"\nLoaded Perch labels: {len(perch_labels)} entries (including non-bird class at idx 0)")

    # Build genus index for proxy matching
    genus_index = build_genus_index(perch_labels)

    # Build mappings
    direct_map: dict[str, int] = {}
    genus_proxy_map: dict[str, list[int]] = {}
    unmapped: list[str] = []

    birdclef_species: list[str] = []
    birdclef_to_scientific: dict[str, str] = {}
    taxa_classes: dict[str, str] = {}

    for _, row in taxonomy.iterrows():
        primary_label = str(row["primary_label"])
        scientific_name = str(row["scientific_name"])
        class_name = str(row["class_name"])

        birdclef_species.append(primary_label)
        birdclef_to_scientific[primary_label] = scientific_name
        taxa_classes[primary_label] = class_name

        # Try direct match by scientific name
        if scientific_name in perch_labels:
            direct_map[primary_label] = perch_labels[scientific_name]
        else:
            # Try genus proxy: find all Perch species in same genus
            genus = scientific_name.split()[0] if " " in scientific_name else ""
            if genus and genus in genus_index:
                genus_proxy_map[primary_label] = sorted(genus_index[genus])
            else:
                unmapped.append(primary_label)

    # Summary
    print("\n=== Mapping Summary ===")
    print(f"  Direct matches:  {len(direct_map)}")
    print(f"  Genus proxies:   {len(genus_proxy_map)}")
    print(f"  Unmapped:        {len(unmapped)}")
    print(f"  Total:           {len(direct_map) + len(genus_proxy_map) + len(unmapped)}")

    # Print genus proxy details
    if genus_proxy_map:
        print("\n  Genus proxy details:")
        for label, indices in genus_proxy_map.items():
            sci_name = birdclef_to_scientific[label]
            genus = sci_name.split()[0]
            print(f"    {label} ({sci_name}) -> genus '{genus}': {len(indices)} Perch species")

    # Print unmapped details
    if unmapped:
        print("\n  Unmapped species:")
        for label in unmapped:
            sci_name = birdclef_to_scientific[label]
            cls = taxa_classes[label]
            print(f"    {label}: {sci_name} ({cls})")

    # Verify expected counts
    n_direct = len(direct_map)
    n_genus = len(genus_proxy_map)
    n_unmapped = len(unmapped)
    total = n_direct + n_genus + n_unmapped
    assert total == len(taxonomy), f"Mapping total {total} != taxonomy size {len(taxonomy)}"

    # Build output
    mapping = {
        "direct_map": direct_map,
        "genus_proxy_map": genus_proxy_map,
        "unmapped": unmapped,
        "birdclef_species": birdclef_species,
        "birdclef_to_scientific": birdclef_to_scientific,
        "taxa_classes": taxa_classes,
        "num_perch_classes": len(perch_labels),
    }

    # Save
    output_path = output_dir / "perch_species_mapping.json"
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nSaved mapping to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
