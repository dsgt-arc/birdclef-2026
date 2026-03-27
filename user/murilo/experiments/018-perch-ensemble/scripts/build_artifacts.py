"""Verify probe artifacts in shared storage and package for Kaggle upload.

Checks that all required probe artifacts exist in the shared probe artifacts
directory (from config paths.probe_artifacts_dir), copies the species mapping
from the local experiment artifacts/ dir, and creates dataset-metadata.json
for Kaggle dataset upload.

Reads:
  - config YAML (paths.probe_artifacts_dir, kaggle slugs)
  - local artifacts/perch_species_mapping.json

Outputs (to shared probe_artifacts_dir):
  - perch_species_mapping.json (copied from local)
  - dataset-metadata.json (for Kaggle upload)
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

REQUIRED_PROBE_FILES = [
    "pca_transformer.pkl",
    "probe_models.pkl",
    "probe_alphas.json",
    "metadata_priors.json",
    "taxa_mapping.json",
    "config.json",
]

SPECIES_MAPPING_FILE = "perch_species_mapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify probe artifacts in shared storage and package for Kaggle upload"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/perch_ensemble_v1.yaml)",
    )
    return parser.parse_args()


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def main() -> None:
    args = parse_args()

    # Resolve config path
    experiment_dir = Path(__file__).resolve().parent.parent
    if args.config is None:
        config_path = experiment_dir / "config" / "perch_ensemble_v1.yaml"
    else:
        config_path = Path(args.config)

    # Load config
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    kaggle = config["kaggle"]

    # Resolve directories
    probe_artifacts_dir = Path(paths["probe_artifacts_dir"])
    local_artifacts_dir = experiment_dir / paths["local_artifacts_dir"]

    print("=" * 60)
    print("Experiment 018: Perch Ensemble Artifact Verification")
    print("=" * 60)
    print(f"\nShared probe artifacts dir: {probe_artifacts_dir}")
    print(f"Local artifacts dir:        {local_artifacts_dir}")

    # ── Verify required probe files in shared dir ───────────────
    print("\n--- Probe Artifacts (shared storage) ---")
    all_ok = True
    total_size = 0

    for filename in REQUIRED_PROBE_FILES:
        filepath = probe_artifacts_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            total_size += size
            print(f"  [OK]      {filename} ({format_size(size)})")
        else:
            print(f"  [MISSING] {filename}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Some required probe artifacts are missing from shared storage!")
        print(f"  Expected location: {probe_artifacts_dir}")
        print("  Run the following scripts first:")
        print(
            "    python scripts/extract_perch_embeddings.py --config config/perch_ensemble_v1.yaml"
        )
        print("    python scripts/train_probes.py --config config/perch_ensemble_v1.yaml")
        raise SystemExit(1)

    # ── Verify and copy species mapping ─────────────────────────
    print("\n--- Species Mapping ---")
    local_mapping_path = local_artifacts_dir / SPECIES_MAPPING_FILE

    if not local_mapping_path.exists():
        print(f"  [MISSING] {local_mapping_path}")
        print(
            "  Run: python scripts/build_species_mapping.py --config config/perch_ensemble_v1.yaml"
        )
        raise SystemExit(1)

    size = local_mapping_path.stat().st_size
    print(f"  [OK] {SPECIES_MAPPING_FILE} ({format_size(size)})")

    # Quick validation
    with open(local_mapping_path) as f:
        mapping = json.load(f)
    n_direct = len(mapping.get("direct_map", {}))
    n_genus = len(mapping.get("genus_proxy_map", {}))
    n_unmapped = len(mapping.get("unmapped", []))
    n_species = len(mapping.get("birdclef_species", []))
    print(
        f"       {n_species} species: {n_direct} direct, {n_genus} genus proxy, {n_unmapped} unmapped"
    )

    # Copy into shared probe artifacts dir
    dest_mapping_path = probe_artifacts_dir / SPECIES_MAPPING_FILE
    if (
        not dest_mapping_path.exists()
        or dest_mapping_path.stat().st_mtime < local_mapping_path.stat().st_mtime
    ):
        shutil.copy2(local_mapping_path, dest_mapping_path)
        print(f"  Copied -> {dest_mapping_path}")
    else:
        print("  Already up-to-date in shared dir")

    total_size += dest_mapping_path.stat().st_size

    # ── Create Kaggle dataset-metadata.json ─────────────────────
    kaggle_slug = f"{kaggle['user']}/{kaggle['perch_artifacts_slug']}"
    dataset_metadata = {
        "id": kaggle_slug,
        "title": "BirdCLEF 2026 Perch Ensemble v1",
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_path = probe_artifacts_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("\n--- Kaggle Dataset Metadata ---")
    print(f"  Created: {metadata_path}")
    print(f"  ID:      {dataset_metadata['id']}")
    print(f"  Title:   {dataset_metadata['title']}")

    # ── Final summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("All artifacts verified and packaged!")
    print(f"{'=' * 60}")
    print(f"\n  Total artifact size: {format_size(total_size)}")

    # List all files in shared probe artifacts dir
    print(f"\nFiles in {probe_artifacts_dir}:")
    for p in sorted(probe_artifacts_dir.iterdir()):
        print(f"  {p.name} ({format_size(p.stat().st_size)})")

    # Print Kaggle upload commands
    print("\nTo upload to Kaggle:")
    print(f"  cd {probe_artifacts_dir}")
    print("  kaggle datasets create -p . --dir-mode zip")
    print("\nOr to update an existing dataset:")
    print(f"  cd {probe_artifacts_dir}")
    print("  kaggle datasets version -p . -m 'Update probe artifacts' --dir-mode zip")


if __name__ == "__main__":
    main()
