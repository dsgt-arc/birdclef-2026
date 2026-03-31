"""Select a diverse species subset for WavTokenizer retrieval experiment.

Picks ~10 species across taxonomic classes (Aves, Amphibia, Insecta, etc.)
that appear in both focal recordings (≥20 files) and soundscape labels.
"""

import json
import random
from pathlib import Path

import pandas as pd
import typer
import yaml

app = typer.Typer()


def _parse_soundscape_species(labels_csv: str) -> set:
    """Extract unique species IDs from soundscape labels (semicolon-separated)."""
    df = pd.read_csv(labels_csv)
    species = set()
    for labels in df["primary_label"].dropna():
        for sp in str(labels).split(";"):
            sp = sp.strip()
            if sp:
                species.add(sp)
    return species


@app.command()
def main(config: str = "configs/exp104.yaml"):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    train_csv = cfg["data"]["train_csv"]
    taxonomy_csv = cfg["data"]["taxonomy_csv"]
    soundscape_labels_csv = cfg["data"]["soundscape_labels_csv"]
    focal_dir = cfg["data"]["focal_dir"]
    n_species = cfg["subset"]["n_species"]
    n_per_species = cfg["subset"]["n_recordings_per_species"]
    seed = cfg["subset"]["seed"]
    output_root = Path(cfg["output"]["root"])

    random.seed(seed)

    # Load metadata
    train_df = pd.read_csv(train_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)
    soundscape_species = _parse_soundscape_species(soundscape_labels_csv)

    # Count focal recordings per species
    focal_counts = train_df["primary_label"].value_counts()

    # Merge with taxonomy for class_name
    # primary_label is the inat taxon ID (as string)
    taxonomy_df["primary_label"] = taxonomy_df["primary_label"].astype(str)
    train_df["primary_label"] = train_df["primary_label"].astype(str)

    species_info = taxonomy_df[["primary_label", "class_name", "common_name", "scientific_name"]].copy()
    species_info["focal_count"] = species_info["primary_label"].map(focal_counts).fillna(0).astype(int)
    species_info["in_soundscape"] = species_info["primary_label"].isin(soundscape_species)

    # Filter: ≥20 focal recordings AND present in soundscape labels
    eligible = species_info[
        (species_info["focal_count"] >= n_per_species) & species_info["in_soundscape"]
    ].copy()

    print(f"Eligible species: {len(eligible)} (≥{n_per_species} focal, in soundscapes)")
    print(f"By class: {eligible['class_name'].value_counts().to_dict()}")

    # Stratified sample across class_name
    selected = []
    class_counts = eligible["class_name"].value_counts()

    # Target allocation proportional to availability, minimum 1 per class
    target_per_class = {}
    for cls in class_counts.index:
        target_per_class[cls] = max(1, round(n_species * class_counts[cls] / len(eligible)))

    # Trim to n_species total
    while sum(target_per_class.values()) > n_species:
        largest = max(target_per_class, key=target_per_class.get)
        target_per_class[largest] -= 1

    print(f"Target per class: {target_per_class}")

    for cls, n_target in target_per_class.items():
        pool = eligible[eligible["class_name"] == cls]
        # Prefer species with moderate counts (20-100), not extremes
        pool = pool.sort_values("focal_count")
        candidates = pool.to_dict("records")
        chosen = random.sample(candidates, min(n_target, len(candidates)))
        selected.extend(chosen)

    # Build output: species_id -> list of file paths
    subset = {}
    for sp in selected:
        sp_id = sp["primary_label"]
        sp_files = train_df[train_df["primary_label"] == sp_id]["filename"].tolist()
        chosen_files = random.sample(sp_files, min(n_per_species, len(sp_files)))
        subset[sp_id] = {
            "class_name": sp["class_name"],
            "common_name": sp["common_name"],
            "scientific_name": sp["scientific_name"],
            "focal_count": sp["focal_count"],
            "files": chosen_files,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "subset.json"
    with open(output_path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"\nSelected {len(subset)} species, {sum(len(v['files']) for v in subset.values())} recordings")
    for sp_id, info in subset.items():
        print(f"  {sp_id}: {info['common_name']} ({info['class_name']}) - {len(info['files'])} files")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    app()
