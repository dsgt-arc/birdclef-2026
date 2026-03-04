#!/usr/bin/env python3
"""Upload Kaggle datasets for BirdCLEF 2025 EfficientNet SED submission.

Creates/updates two Kaggle datasets:
- Dependencies: wheel files (onnxruntime, librosa, etc.) — reuses existing dataset
- Weights: ONNX model, species_list.json, mel_config.json, kaggle_inference.py

Usage:
    python scripts/upload_kaggle.py [--dry-run] [--kaggle-user USERNAME]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Default configuration -- override with --kaggle-user
DEFAULT_KAGGLE_USER = "acmiyaguchi"

DATASETS = {
    "birdclef-2025-deps": "artifacts/dist/dependencies",
    "birdclef-2025-weights": "artifacts/weights",
}


def get_experiment_dir() -> Path:
    """Get the experiment directory path."""
    return Path(__file__).resolve().parent.parent


def create_metadata(slug: str, kaggle_user: str, path: Path) -> None:
    """Create dataset-metadata.json for Kaggle."""
    meta = {
        "title": slug,
        "id": f"{kaggle_user}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = path / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Created: {meta_path}")


def run_kaggle_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a kaggle CLI command. Returns True if successful, False if dataset already exists."""
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        print(result.stdout)
        if "already in use" in output or "already exists" in output:
            return False
        return True
    except subprocess.CalledProcessError as e:
        combined = e.stdout + e.stderr
        if "already exists" in combined or "already in use" in combined or "403" in combined:
            return False
        print(f"  Error: {e.stderr}")
        raise


def stage_weights() -> None:
    """Stage weight artifacts: copy inference script into weights directory."""
    experiment_dir = get_experiment_dir()
    weights_dir = experiment_dir / "artifacts" / "weights"

    inference_script = experiment_dir / "scripts" / "kaggle_inference.py"
    if inference_script.exists() and weights_dir.exists():
        shutil.copy2(inference_script, weights_dir / "kaggle_inference.py")
        print(f"  Staged: {inference_script.name} -> {weights_dir}")
    else:
        if not weights_dir.exists():
            print(f"  WARNING: {weights_dir} not found, run export-onnx first")
        if not inference_script.exists():
            print(f"  WARNING: {inference_script} not found")


def upload_dataset(slug: str, rel_path: str, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload or update a single dataset."""
    experiment_dir = get_experiment_dir()
    path = experiment_dir / rel_path

    if not path.exists():
        print(f"  WARNING: {path} does not exist, skipping")
        return

    files = [f for f in path.glob("*") if f.name != "dataset-metadata.json"]
    if not files:
        print(f"  WARNING: {path} appears empty, skipping")
        return

    print(f"\n{'='*50}")
    print(f"Uploading: {kaggle_user}/{slug}")
    print(f"  Source: {path}")
    print(f"  Files: {len(files)}")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name} ({size_mb:.1f} MB)")

    create_metadata(slug, kaggle_user, path)

    create_cmd = ["kaggle", "datasets", "create", "-p", str(path), "-r", "zip"]
    try:
        if run_kaggle_cmd(create_cmd, dry_run):
            print(f"  Created new dataset: {kaggle_user}/{slug}")
            return
    except subprocess.CalledProcessError:
        pass

    print(f"  Dataset exists, creating new version...")
    version_cmd = [
        "kaggle", "datasets", "version",
        "-p", str(path),
        "-m", "Auto-update from HPC build",
        "-r", "zip",
    ]
    run_kaggle_cmd(version_cmd, dry_run)
    print(f"  Updated dataset: {kaggle_user}/{slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Kaggle datasets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without uploading",
    )
    parser.add_argument(
        "--kaggle-user",
        type=str,
        default=DEFAULT_KAGGLE_USER,
        help=f"Kaggle username (default: {DEFAULT_KAGGLE_USER})",
    )
    args = parser.parse_args()

    experiment_dir = get_experiment_dir()

    print("Kaggle Dataset Uploader")
    print("=" * 50)
    print(f"User: {args.kaggle_user}")
    print(f"Experiment: {experiment_dir}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No uploads will be performed ***\n")

    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("ERROR: kaggle CLI not configured. Set up ~/.kaggle/kaggle.json")
        sys.exit(1)

    stage_weights()

    for slug, rel_path in DATASETS.items():
        upload_dataset(slug, rel_path, args.kaggle_user, args.dry_run)

    print("\n" + "=" * 50)
    print("Done!")
    print(f"\nTo create a Kaggle kernel, use these datasets as inputs:")
    for slug in DATASETS:
        print(f"  - {args.kaggle_user}/{slug}")


if __name__ == "__main__":
    main()
