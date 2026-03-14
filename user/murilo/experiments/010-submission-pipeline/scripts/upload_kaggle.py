#!/usr/bin/env python3
"""Upload model weights to Kaggle datasets and push submission kernels.

Creates/updates Kaggle datasets with ONNX model weights and pushes the
unified submission kernel with model-specific metadata.

Usage:
    # Upload weights for a specific model
    python scripts/upload_kaggle.py --model eat
    python scripts/upload_kaggle.py --model convnext
    python scripts/upload_kaggle.py --model protoclr

    # Upload all models
    python scripts/upload_kaggle.py --model all

    # Push kernel for a model
    python scripts/upload_kaggle.py --model eat --push-kernel

    # Dry run
    python scripts/upload_kaggle.py --model eat --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_KAGGLE_USER = "murilogustineli"

MODELS = {
    "eat": {
        "slug": "birdclef-2025-eat-weights",
        "weights_dir": "artifacts/weights/eat",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat.json",
    },
    "convnext": {
        "slug": "birdclef-2025-convnext-weights",
        "weights_dir": "artifacts/weights/convnext",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext.json",
    },
    "protoclr": {
        "slug": "birdclef-2025-protoclr-weights",
        "weights_dir": "artifacts/weights/protoclr",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr.json",
    },
}


def get_experiment_dir() -> Path:
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


def run_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a command. Returns True if successful."""
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        output = result.stdout + result.stderr
        if "already in use" in output or "already exists" in output:
            return False
        return True
    except subprocess.CalledProcessError as e:
        combined = e.stdout + e.stderr
        if "already exists" in combined or "403" in combined:
            return False
        if e.stdout:
            print(f"  stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"  stderr: {e.stderr.strip()}")
        raise


def upload_weights(model_name: str, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload model weights to Kaggle dataset."""
    model_cfg = MODELS[model_name]
    experiment_dir = get_experiment_dir()
    weights_dir = experiment_dir / model_cfg["weights_dir"]
    slug = model_cfg["slug"]

    if not weights_dir.exists():
        print(f"  WARNING: {weights_dir} does not exist. Run export first.")
        return

    # Copy species_mapping.json into model weights dir
    mapping_src = experiment_dir / "artifacts" / "weights" / "species_mapping.json"
    mapping_dst = weights_dir / "species_mapping.json"
    if mapping_src.exists() and not mapping_dst.exists():
        shutil.copy2(mapping_src, mapping_dst)
        print(f"  Copied species_mapping.json to {weights_dir}")

    files = [f for f in weights_dir.glob("*") if f.name != "dataset-metadata.json"]
    if not files:
        print(f"  WARNING: {weights_dir} is empty. Run export first.")
        return

    print(f"\n{'='*50}")
    print(f"Uploading: {kaggle_user}/{slug}")
    print(f"  Source: {weights_dir}")
    print("  Files:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name} ({size_mb:.1f} MB)")

    create_metadata(slug, kaggle_user, weights_dir)

    # Try create, then version
    try:
        if run_cmd(["kaggle", "datasets", "create", "-p", str(weights_dir), "-r", "zip"], dry_run):
            print(f"  Created new dataset: {kaggle_user}/{slug}")
            return
    except subprocess.CalledProcessError:
        pass

    print("  Dataset exists, creating new version...")
    run_cmd(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(weights_dir),
            "-m",
            "Auto-update from HPC export",
            "-r",
            "zip",
        ],
        dry_run,
    )
    print(f"  Updated: {kaggle_user}/{slug}")


def push_kernel(model_name: str, dry_run: bool = False) -> None:
    """Push submission kernel to Kaggle."""
    model_cfg = MODELS[model_name]
    experiment_dir = get_experiment_dir()
    kernel_dir = experiment_dir / "kaggle_kernel"
    metadata_src = experiment_dir / model_cfg["kernel_metadata"]

    if not metadata_src.exists():
        print(f"  WARNING: {metadata_src} not found")
        return

    # Copy model-specific metadata as kernel-metadata.json
    metadata_dst = kernel_dir / "kernel-metadata.json"
    shutil.copy2(metadata_src, metadata_dst)
    print(f"  Using metadata: {metadata_src.name}")

    run_cmd(["kaggle", "kernels", "push", "-p", str(kernel_dir)], dry_run)
    print(f"  Kernel pushed for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload Kaggle datasets and kernels")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["eat", "convnext", "protoclr", "all"],
        help="Model to upload",
    )
    parser.add_argument("--kaggle-user", type=str, default=DEFAULT_KAGGLE_USER)
    parser.add_argument("--push-kernel", action="store_true", help="Also push the Kaggle kernel")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Check kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: kaggle CLI not found or not configured")
        sys.exit(1)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        upload_weights(model_name, args.kaggle_user, args.dry_run)
        if args.push_kernel:
            push_kernel(model_name, args.dry_run)

    print(f"\nDone! Uploaded {len(models)} model(s).")


if __name__ == "__main__":
    main()
