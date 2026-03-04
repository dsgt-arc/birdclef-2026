"""Build BirdNET/Perch → BirdCLEF-206 species index mappings.

Maps the raw logit column indices from BirdNET (6522 species) and Perch
(10932 species) predict parquets to the 206 BirdCLEF species codes.

BirdNET labels: "Genus species_Common Name" format, matched via scientific_name
Perch labels: eBird codes (ebird2021), matched directly to primary_label

CLI:
    weaksup-build-mapping \\
        --taxonomy-csv ~/scratch/birdclef-2026/data/birdclef-2025/taxonomy.csv \\
        --perch-label-csv /tmp/tfhub_modules/.../assets/label.csv \\
        --output-dir ~/scratch/birdclef-2026/outputs/exp013/mappings
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

BIRDNET_LABELS_URL = (
    "https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/"
    "v1.3.1/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt"
)

DEFAULT_TAXONOMY = Path("~/scratch/birdclef-2026/data/birdclef-2025/taxonomy.csv").expanduser()
DEFAULT_PERCH_LABELS = Path(
    "/tmp/tfhub_modules/50cbe9e746f7b3e38819b8f3dc1f73e682ceca90/assets/label.csv"
)
DEFAULT_OUTPUT_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/mappings").expanduser()


def _load_taxonomy(taxonomy_csv: Path) -> list[dict[str, str]]:
    """Load taxonomy.csv rows as list of dicts."""
    with open(taxonomy_csv) as f:
        return list(csv.DictReader(f))


def build_perch_mapping(
    perch_label_csv: Path,
    taxonomy_rows: list[dict[str, str]],
) -> tuple[dict[str, int], list[str]]:
    """Map BirdCLEF primary_label → Perch column index for birds.

    Perch label.csv has header "ebird2021" + one eBird code per line.
    Column index = line number - 1 (0-indexed after header).

    Returns:
        mapping: {birdclef_species: perch_column_index} for matched birds
        missing: list of bird species not found in Perch labels
    """
    # Load Perch eBird codes with their 0-based indices
    perch_code_to_idx: dict[str, int] = {}
    with open(perch_label_csv) as f:
        header = f.readline().strip()
        assert header == "ebird2021", f"Unexpected Perch label header: {header!r}"
        for idx, line in enumerate(f):
            code = line.strip()
            if code:
                perch_code_to_idx[code] = idx

    # Match BirdCLEF bird species to Perch codes
    bird_species = [
        r["primary_label"] for r in taxonomy_rows if r["class_name"] == "Aves"
    ]

    mapping: dict[str, int] = {}
    missing: list[str] = []
    for sp in bird_species:
        if sp in perch_code_to_idx:
            mapping[sp] = perch_code_to_idx[sp]
        else:
            missing.append(sp)

    return mapping, missing


def build_birdnet_mapping(
    birdnet_labels: list[str],
    taxonomy_rows: list[dict[str, str]],
) -> tuple[dict[str, int], list[str]]:
    """Map BirdCLEF primary_label → BirdNET column index for birds.

    BirdNET labels are "Genus species_Common Name" format.
    Match "Genus species" against taxonomy.csv scientific_name.

    Returns:
        mapping: {birdclef_species: birdnet_column_index} for matched birds
        missing: list of bird species not found in BirdNET labels
    """
    # Build scientific_name → BirdNET index lookup
    sciname_to_idx: dict[str, int] = {}
    for idx, label in enumerate(birdnet_labels):
        # "Genus species_Common Name" → "Genus species"
        sciname = label.split("_")[0].strip()
        sciname_to_idx[sciname] = idx

    # Match BirdCLEF bird species
    bird_rows = [r for r in taxonomy_rows if r["class_name"] == "Aves"]

    mapping: dict[str, int] = {}
    missing: list[str] = []
    for row in bird_rows:
        sp = row["primary_label"]
        sciname = row["scientific_name"]
        if sciname in sciname_to_idx:
            mapping[sp] = sciname_to_idx[sciname]
        else:
            missing.append(sp)

    return mapping, missing


def _download_birdnet_labels(url: str) -> list[str]:
    """Download BirdNET labels.txt from GitHub."""
    console.print(f"Downloading BirdNET labels from {url}")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    console.print(f"  Downloaded {len(labels)} BirdNET labels")
    return labels


@app.command()
def build_mapping(
    taxonomy_csv: Annotated[
        Path, typer.Option(help="Path to taxonomy.csv")
    ] = DEFAULT_TAXONOMY,
    perch_label_csv: Annotated[
        Path, typer.Option(help="Path to Perch label.csv (TF Hub cache)")
    ] = DEFAULT_PERCH_LABELS,
    birdnet_label_url: Annotated[
        str, typer.Option(help="URL or local path for BirdNET labels.txt")
    ] = BIRDNET_LABELS_URL,
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for mapping JSON")
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Build and save BirdNET/Perch → BirdCLEF species mappings."""
    console.rule("[bold]Building species mappings")

    taxonomy_rows = _load_taxonomy(taxonomy_csv)
    n_birds = sum(1 for r in taxonomy_rows if r["class_name"] == "Aves")
    console.print(f"Taxonomy: {len(taxonomy_rows)} species, {n_birds} birds")

    # Perch mapping
    console.rule("[bold]Perch mapping")
    if not perch_label_csv.exists():
        console.print(f"[red]Perch label.csv not found at {perch_label_csv}[/red]")
        raise typer.Exit(1)
    perch_map, perch_missing = build_perch_mapping(perch_label_csv, taxonomy_rows)
    console.print(f"  Matched: {len(perch_map)}/{n_birds} birds")
    if perch_missing:
        console.print(f"  Missing: {perch_missing}")

    # BirdNET mapping
    console.rule("[bold]BirdNET mapping")
    if Path(birdnet_label_url).exists():
        with open(birdnet_label_url) as f:
            birdnet_labels = [line.strip() for line in f if line.strip()]
    else:
        birdnet_labels = _download_birdnet_labels(birdnet_label_url)
    birdnet_map, birdnet_missing = build_birdnet_mapping(birdnet_labels, taxonomy_rows)
    console.print(f"  Matched: {len(birdnet_map)}/{n_birds} birds")
    if birdnet_missing:
        console.print(f"  Missing: {birdnet_missing}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "species_mapping.json"
    result = {
        "perch": perch_map,
        "birdnet": birdnet_map,
        "perch_missing": perch_missing,
        "birdnet_missing": birdnet_missing,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    console.print(f"\nSaved to {output_path}")
    console.print(f"  Perch: {len(perch_map)} entries, {len(perch_missing)} missing")
    console.print(f"  BirdNET: {len(birdnet_map)} entries, {len(birdnet_missing)} missing")


if __name__ == "__main__":
    app()
