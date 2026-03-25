#!/usr/bin/env python3
"""Build inference prior artifacts for pp2 post-processing.

Reads taxonomy.csv + train_soundscapes_labels.csv + species_mapping.json
and outputs two JSON files for bundling into pp2 weights:

  taxa_mapping.json    — species indices grouped by taxa class
  metadata_priors.json — site/hour/species occurrence priors from labeled soundscapes

Usage:
    python scripts/build_inference_priors.py \
        --data-dir /storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026 \
        --species-mapping artifacts/weights/species_mapping_2026_234.json \
        --output-dir artifacts/weights/effnet-b0-sed-v1-10s-pp2
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def build_taxa_mapping(taxonomy_csv: Path, species_mapping_json: Path) -> dict:
    """Build taxa_mapping: {taxa_class: [model_output_indices]}."""
    # taxonomy.csv: primary_label -> class_name
    label_to_taxa = {}
    with open(taxonomy_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_to_taxa[row["primary_label"]] = row["class_name"].lower()

    # species_mapping.json: idx -> primary_label
    with open(species_mapping_json) as f:
        mapping = json.load(f)
    idx_to_name = mapping.get("idx_to_name_2026") or mapping["idx_2025_to_name_2026"]

    # Cross-reference: taxa_class -> [indices]
    taxa_mapping = defaultdict(list)
    for idx_str, primary_label in idx_to_name.items():
        taxa = label_to_taxa.get(primary_label, "unknown")
        taxa_mapping[taxa].append(int(idx_str))

    # Sort indices within each class
    result = {k: sorted(v) for k, v in taxa_mapping.items()}
    return result


def build_metadata_priors(
    labels_csv: Path, soundscapes_dir: Path, species_mapping_json: Path, laplace_alpha: float = 0.01
) -> dict:
    """Build site/hour/species occurrence priors from labeled soundscapes."""
    # Get all species codes
    with open(species_mapping_json) as f:
        mapping = json.load(f)
    idx_to_name = mapping.get("idx_to_name_2026") or mapping["idx_2025_to_name_2026"]
    all_species = set(idx_to_name.values())
    n_species = len(all_species)

    # Parse labeled soundscapes
    # Format: filename,start,end,primary_label
    # primary_label is semicolon-separated
    site_hour_species = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    site_hour_total = defaultdict(lambda: defaultdict(int))
    site_species = defaultdict(lambda: defaultdict(int))
    site_total = defaultdict(int)
    hour_species = defaultdict(lambda: defaultdict(int))
    hour_total = defaultdict(int)
    global_species = defaultdict(int)
    global_total = 0

    seen_files = set()

    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            labels = row["primary_label"].split(";")

            # Parse metadata from filename
            # BC2026_Train_0039_S22_20211231_201500.ogg
            parts = Path(filename).stem.split("_")
            if len(parts) < 6:
                continue
            site = parts[3]  # e.g. "S22"
            hour = parts[5][:2]  # e.g. "20"

            seen_files.add(filename)

            for sp in labels:
                sp = sp.strip()
                if sp and sp in all_species:
                    site_hour_species[site][hour][sp] += 1
                    site_species[site][sp] += 1
                    hour_species[hour][sp] += 1
                    global_species[sp] += 1

            site_hour_total[site][hour] += 1
            site_total[site] += 1
            hour_total[hour] += 1
            global_total += 1

    # Compute probabilities with Laplace smoothing
    def compute_probs(count_dict, total, alpha=laplace_alpha):
        probs = {}
        for sp in all_species:
            count = count_dict.get(sp, 0)
            probs[sp] = (count + alpha) / (total + alpha * n_species)
        return probs

    # Site+hour priors
    site_hour_priors = {}
    for site in site_hour_species:
        site_hour_priors[site] = {}
        for hour in site_hour_species[site]:
            total = site_hour_total[site][hour]
            site_hour_priors[site][hour] = compute_probs(site_hour_species[site][hour], total)

    # Site-only priors
    site_priors = {}
    for site in site_species:
        site_priors[site] = compute_probs(site_species[site], site_total[site])

    # Hour-only priors
    hour_priors = {}
    for hour in hour_species:
        hour_priors[hour] = compute_probs(hour_species[hour], hour_total[hour])

    # Global priors
    global_probs = compute_probs(global_species, global_total)

    return {
        "site_hour_species": site_hour_priors,
        "site_species": site_priors,
        "hour_species": hour_priors,
        "global_species": global_probs,
        "sites": sorted(site_total.keys()),
        "hours": sorted(hour_total.keys()),
        "n_soundscapes": len(seen_files),
        "n_segments": global_total,
        "laplace_alpha": laplace_alpha,
    }


def main():
    parser = argparse.ArgumentParser(description="Build inference priors for pp2")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026",
    )
    parser.add_argument(
        "--species-mapping",
        type=str,
        default="artifacts/weights/species_mapping_2026_234.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/weights/effnet-b0-sed-v1-10s-pp2",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    species_mapping_path = script_dir / args.species_mapping
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_csv = data_dir / "taxonomy.csv"
    labels_csv = data_dir / "train_soundscapes_labels.csv"
    soundscapes_dir = data_dir / "train_soundscapes"

    print(f"Data dir: {data_dir}")
    print(f"Species mapping: {species_mapping_path}")
    print(f"Output dir: {output_dir}")

    # Build taxa mapping
    print("\nBuilding taxa mapping...")
    taxa_mapping = build_taxa_mapping(taxonomy_csv, species_mapping_path)
    for taxa, indices in sorted(taxa_mapping.items()):
        print(f"  {taxa}: {len(indices)} species")

    taxa_path = output_dir / "taxa_mapping.json"
    with open(taxa_path, "w") as f:
        json.dump(taxa_mapping, f, indent=2)
    print(f"Saved: {taxa_path}")

    # Build metadata priors
    print("\nBuilding metadata priors...")
    priors = build_metadata_priors(labels_csv, soundscapes_dir, species_mapping_path)
    print(f"  Sites: {priors['sites']}")
    print(f"  Hours: {priors['hours']}")
    print(f"  Soundscapes: {priors['n_soundscapes']}")
    print(f"  Segments: {priors['n_segments']}")

    priors_path = output_dir / "metadata_priors.json"
    with open(priors_path, "w") as f:
        json.dump(priors, f, indent=2)
    print(f"Saved: {priors_path} ({priors_path.stat().st_size / 1024:.1f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
