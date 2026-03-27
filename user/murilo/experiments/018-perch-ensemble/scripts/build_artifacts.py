"""Verify and package artifacts for Kaggle upload.

Checks that all required artifacts exist in artifacts/perch_probes/
and creates dataset-metadata.json for Kaggle dataset upload.
"""

import argparse
import json
from pathlib import Path

REQUIRED_FILES = [
    "pca_transformer.pkl",
    "probe_models.pkl",
    "probe_alphas.json",
    "metadata_priors.json",
    "taxa_mapping.json",
    "config.json",
]

SPECIES_MAPPING_FILE = "perch_species_mapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and package artifacts for Kaggle upload")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Path to artifacts directory (default: artifacts/ relative to experiment dir)",
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

    # Resolve paths
    if args.artifacts_dir is None:
        experiment_dir = Path(__file__).resolve().parent.parent
        artifacts_dir = experiment_dir / "artifacts"
    else:
        artifacts_dir = Path(args.artifacts_dir)

    probes_dir = artifacts_dir / "perch_probes"

    print("=" * 60)
    print("Experiment 018: Perch Ensemble Artifact Verification")
    print("=" * 60)
    print(f"\nArtifacts directory: {artifacts_dir}")
    print(f"Probes directory:   {probes_dir}")

    # Check species mapping in artifacts/
    print("\n--- Species Mapping ---")
    mapping_path = artifacts_dir / SPECIES_MAPPING_FILE
    if mapping_path.exists():
        size = mapping_path.stat().st_size
        print(f"  [OK] {SPECIES_MAPPING_FILE} ({format_size(size)})")

        # Quick validation
        with open(mapping_path) as f:
            mapping = json.load(f)
        n_direct = len(mapping.get("direct_map", {}))
        n_genus = len(mapping.get("genus_proxy_map", {}))
        n_unmapped = len(mapping.get("unmapped", []))
        n_species = len(mapping.get("birdclef_species", []))
        print(
            f"       {n_species} species: {n_direct} direct, {n_genus} genus proxy, {n_unmapped} unmapped"
        )
    else:
        print(f"  [MISSING] {SPECIES_MAPPING_FILE}")

    # Check probe artifacts in artifacts/perch_probes/
    print("\n--- Probe Artifacts ---")
    all_ok = True
    total_size = 0

    for filename in REQUIRED_FILES:
        filepath = probes_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            total_size += size
            print(f"  [OK] {filename} ({format_size(size)})")
        else:
            print(f"  [MISSING] {filename}")
            all_ok = False

    # Also include species mapping in the upload
    if mapping_path.exists():
        total_size += mapping_path.stat().st_size

    print(f"\n  Total artifact size: {format_size(total_size)}")

    if not all_ok:
        print("\n[ERROR] Some required artifacts are missing!")
        print("  Run the following scripts first:")
        print("    python scripts/build_species_mapping.py")
        print("    python scripts/extract_perch_embeddings.py")
        print("    python scripts/train_probes.py")
        raise SystemExit(1)

    # Copy species mapping into probes dir for Kaggle upload
    if mapping_path.exists():
        import shutil

        dest = probes_dir / SPECIES_MAPPING_FILE
        if not dest.exists() or dest.stat().st_mtime < mapping_path.stat().st_mtime:
            shutil.copy2(mapping_path, dest)
            print(f"\n  Copied {SPECIES_MAPPING_FILE} -> perch_probes/")

    # Create Kaggle dataset-metadata.json
    dataset_metadata = {
        "id": "murilogustineli/birdclef-2026-perch-ensemble-v1",
        "title": "BirdCLEF 2026 Perch Ensemble v1",
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_path = probes_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("\n--- Kaggle Dataset Metadata ---")
    print(f"  Created: {metadata_path}")
    print(f"  ID: {dataset_metadata['id']}")
    print(f"  Title: {dataset_metadata['title']}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("All artifacts verified and packaged!")
    print(f"{'=' * 60}")

    print("\nTo upload to Kaggle:")
    print(f"  cd {probes_dir}")
    print("  kaggle datasets create -p . --dir-mode zip")
    print("\nOr use the upload script:")
    print(f"  python scripts/upload_kaggle.py --artifact-dir {probes_dir}")

    # List all files in probes dir
    print(f"\nFiles in {probes_dir}:")
    for p in sorted(probes_dir.iterdir()):
        print(f"  {p.name} ({format_size(p.stat().st_size)})")


if __name__ == "__main__":
    main()
