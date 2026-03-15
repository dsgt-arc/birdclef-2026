#!/usr/bin/env python3
"""Upload model weights to Kaggle datasets and push submission kernels.

Creates/updates Kaggle datasets with ONNX model weights and pushes the
unified submission kernel with model-specific metadata. Supports both
BirdCLEF 2025-trained and 2026-trained models.

Model names always include the training data year for clarity:
  - protoclr-2025, eat-2025, convnext-2025  (exps 006/007/009)
  - protoclr-2026, eat-2026, convnext-2026  (exp 011)

Usage:
    # Upload 2025-trained models
    python scripts/upload_kaggle.py --model protoclr-2025
    python scripts/upload_kaggle.py --model all-2025

    # Upload 2026-trained models (exp 011)
    python scripts/upload_kaggle.py --model protoclr-2026
    python scripts/upload_kaggle.py --model all-2026

    # Upload + push kernel
    python scripts/upload_kaggle.py --model protoclr-2026 --push-kernel

    # Dry run
    python scripts/upload_kaggle.py --model protoclr-2026 --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_KAGGLE_USER = "murilogustineli"

# Separate deps dataset (onnxruntime wheels for offline Kaggle install)
DEPS_SLUG = "birdclef-deps"
DEPS_DIR = "artifacts/dist/dependencies"

MODELS = {
    # Models trained on BirdCLEF 2025 data (exps 006/007/009)
    "eat-2025": {
        "slug": "birdclef-2025-eat-weights",
        "weights_dir": "artifacts/weights/eat-2025",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2025.json",
    },
    "convnext-2025": {
        "slug": "birdclef-2025-convnext-weights",
        "weights_dir": "artifacts/weights/convnext-2025",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext-2025.json",
    },
    "protoclr-2025": {
        "slug": "birdclef-2025-protoclr-weights",
        "weights_dir": "artifacts/weights/protoclr-2025",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2025.json",
    },
    # Models trained on BirdCLEF 2026 focal data (exp 011)
    "eat-2026": {
        "slug": "birdclef-2026-eat-weights",
        "weights_dir": "artifacts/weights/eat-2026",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2026.json",
    },
    "convnext-2026": {
        "slug": "birdclef-2026-convnext-weights",
        "weights_dir": "artifacts/weights/convnext-2026",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext-2026.json",
    },
    "protoclr-2026": {
        "slug": "birdclef-2026-protoclr-weights",
        "weights_dir": "artifacts/weights/protoclr-2026",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2026.json",
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


def upload_dataset(slug: str, path: Path, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload or update a single Kaggle dataset."""
    if not path.exists():
        print(f"  WARNING: {path} does not exist, skipping")
        return

    files = [f for f in path.glob("*") if f.name != "dataset-metadata.json"]
    if not files:
        print(f"  WARNING: {path} is empty, skipping")
        return

    print(f"\n{'='*50}")
    print(f"Uploading: {kaggle_user}/{slug}")
    print(f"  Source: {path}")
    print("  Files:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name} ({size_mb:.1f} MB)")

    create_metadata(slug, kaggle_user, path)

    # Try create, then version
    try:
        if run_cmd(["kaggle", "datasets", "create", "-p", str(path), "-r", "zip"], dry_run):
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
            str(path),
            "-m",
            "Auto-update from HPC export",
            "-r",
            "zip",
        ],
        dry_run,
    )
    print(f"  Updated: {kaggle_user}/{slug}")


def upload_weights(model_name: str, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload model weights to Kaggle dataset."""
    model_cfg = MODELS[model_name]
    experiment_dir = get_experiment_dir()
    weights_dir = experiment_dir / model_cfg["weights_dir"]

    if not weights_dir.exists():
        print(f"  WARNING: {weights_dir} does not exist. Run export first.")
        return

    # Copy species_mapping.json into model weights dir
    # Use 2026 mapping for 2026-trained models, 2025 mapping otherwise
    if model_name.endswith("-2026"):
        mapping_src = experiment_dir / "artifacts" / "weights" / "species_mapping_2026.json"
    else:
        mapping_src = experiment_dir / "artifacts" / "weights" / "species_mapping.json"
    mapping_dst = weights_dir / "species_mapping.json"
    if mapping_src.exists():
        shutil.copy2(mapping_src, mapping_dst)
        print(f"  Copied {mapping_src.name} to {weights_dir}")
    elif not mapping_dst.exists():
        print(f"  WARNING: {mapping_src} not found, species_mapping.json missing")

    upload_dataset(model_cfg["slug"], weights_dir, kaggle_user, dry_run)


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
        choices=list(MODELS.keys()) + ["all", "all-2025", "all-2026"],
        help="Model to upload (e.g., protoclr-2025, protoclr-2026, all-2025, all-2026)",
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

    experiment_dir = get_experiment_dir()
    if args.model == "all":
        models = list(MODELS.keys())
    elif args.model == "all-2025":
        models = [k for k in MODELS if k.endswith("-2025")]
    elif args.model == "all-2026":
        models = [k for k in MODELS if k.endswith("-2026")]
    else:
        models = [args.model]

    # Upload deps dataset (shared across all models)
    deps_path = experiment_dir / DEPS_DIR
    upload_dataset(DEPS_SLUG, deps_path, args.kaggle_user, args.dry_run)

    # Upload model weights + push kernels
    for model_name in models:
        upload_weights(model_name, args.kaggle_user, args.dry_run)
        if args.push_kernel:
            push_kernel(model_name, args.dry_run)

    print(f"\nDone! Uploaded deps + {len(models)} model(s).")


if __name__ == "__main__":
    main()
