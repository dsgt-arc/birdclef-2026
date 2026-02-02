#!/usr/bin/env python3
"""Upload Kaggle datasets for BirdCLEF submission.

Creates/updates three Kaggle datasets from the built artifacts:
- Dependencies (wheel files)
- Code (source bundles)
- Weights (model checkpoints)

Usage:
    python scripts/upload_datasets.py [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Configuration
KAGGLE_USER = "cheungus"
DATASETS = {
    "birdclef-2026-deps": "dependencies",
    "birdclef-2026-code": "code",
    "birdclef-2026-weights": "weights",
}


def get_dist_dir() -> Path:
    """Get the dist directory path."""
    script_dir = Path(__file__).parent
    experiment_dir = script_dir.parent
    return experiment_dir / "artifacts" / "dist"


def create_metadata(slug: str, path: Path) -> None:
    """Create dataset-metadata.json for Kaggle."""
    meta = {
        "title": slug,
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = path / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Created: {meta_path}")


def run_kaggle_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a kaggle CLI command."""
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stderr or "403" in e.stderr:
            return False  # Dataset exists, need to version
        print(f"  Error: {e.stderr}")
        raise


def upload_dataset(slug: str, subdir: str, dry_run: bool = False) -> None:
    """Upload or update a single dataset."""
    dist_dir = get_dist_dir()
    path = dist_dir / subdir

    if not path.exists():
        print(f"  WARNING: {path} does not exist, skipping")
        return

    files = list(path.glob("*"))
    if len(files) <= 1:  # Only metadata or empty
        print(f"  WARNING: {path} appears empty, skipping")
        return

    print(f"\n{'='*50}")
    print(f"Uploading: {slug}")
    print(f"  Source: {path}")
    print(f"  Files: {len(files)}")

    # Create metadata
    create_metadata(slug, path)

    # Try to create new dataset
    create_cmd = ["kaggle", "datasets", "create", "-p", str(path), "-r", "zip"]
    try:
        if run_kaggle_cmd(create_cmd, dry_run):
            print(f"  ✓ Created new dataset: {KAGGLE_USER}/{slug}")
            return
    except subprocess.CalledProcessError:
        pass

    # Dataset exists, create new version
    print(f"  Dataset exists, creating new version...")
    version_cmd = [
        "kaggle", "datasets", "version",
        "-p", str(path),
        "-m", "Auto-update from HPC build",
        "-r", "zip",
    ]
    run_kaggle_cmd(version_cmd, dry_run)
    print(f"  ✓ Updated dataset: {KAGGLE_USER}/{slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Kaggle datasets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without uploading",
    )
    args = parser.parse_args()

    print("Kaggle Dataset Uploader")
    print("=" * 50)
    print(f"User: {KAGGLE_USER}")
    print(f"Dist dir: {get_dist_dir()}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No uploads will be performed ***\n")

    # Check kaggle CLI is available
    try:
        subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("ERROR: kaggle CLI not configured. Set up ~/.kaggle/kaggle.json")
        sys.exit(1)

    # Upload each dataset
    for slug, subdir in DATASETS.items():
        upload_dataset(slug, subdir, args.dry_run)

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
