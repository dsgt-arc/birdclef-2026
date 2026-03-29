#!/usr/bin/env python3
"""Upload Perch ensemble artifacts to Kaggle and push submission kernels.

Creates/updates the Kaggle dataset with probe artifacts (PCA, LogReg models,
priors, species mapping) and pushes the submission kernel.

Usage:
    # Upload artifacts only
    python scripts/upload_kaggle.py

    # Upload artifacts + push Perch-only kernel (Phase A)
    python scripts/upload_kaggle.py --push-kernel --perch-only

    # Upload artifacts + push ensemble kernel (Phase B)
    python scripts/upload_kaggle.py --push-kernel

    # Dry run
    python scripts/upload_kaggle.py --push-kernel --perch-only --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a command. Returns True if successful."""
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print(result.stdout.strip())
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


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def upload_dataset(slug: str, path: Path, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload or update a Kaggle dataset."""
    if not path.exists():
        print(f"  ERROR: {path} does not exist")
        sys.exit(1)

    files = [f for f in sorted(path.glob("*")) if f.name != "dataset-metadata.json"]
    if not files:
        print(f"  ERROR: {path} is empty")
        sys.exit(1)

    full_slug = f"{kaggle_user}/{slug}"
    print(f"\n{'=' * 50}")
    print(f"Uploading dataset: {full_slug}")
    print(f"  Source: {path}")
    print("  Files:")
    total = 0
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"    {f.name} ({format_size(size)})")
    print(f"  Total: {format_size(total)}")

    # Ensure dataset-metadata.json exists
    meta_path = path / "dataset-metadata.json"
    meta = {
        "id": full_slug,
        "title": slug.replace("-", " ").title(),
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Try create first, then version update
    try:
        if run_cmd(["kaggle", "datasets", "create", "-p", str(path), "-r", "zip"], dry_run):
            print(f"  Created new dataset: {full_slug}")
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
            "Update probe artifacts",
            "-r",
            "zip",
        ],
        dry_run,
    )
    print(f"  Updated: {full_slug}")


def push_kernel(kernel_dir: Path, metadata_file: str, dry_run: bool = False) -> None:
    """Push a submission kernel to Kaggle."""
    metadata_src = kernel_dir / metadata_file
    metadata_dst = kernel_dir / "kernel-metadata.json"

    if not metadata_src.exists():
        print(f"  ERROR: {metadata_src} not found")
        sys.exit(1)

    # Copy the chosen metadata as kernel-metadata.json (what kaggle CLI reads)
    shutil.copy2(metadata_src, metadata_dst)

    # Read kernel slug for display
    with open(metadata_src) as f:
        meta = json.load(f)
    kernel_slug = meta.get("id", "unknown")

    print(f"\n{'=' * 50}")
    print(f"Pushing kernel: {kernel_slug}")
    print(f"  Metadata: {metadata_file}")
    print(f"  Code: {meta.get('code_file', 'submission.py')}")

    run_cmd(["kaggle", "kernels", "push", "-p", str(kernel_dir)], dry_run)
    print(f"  Kernel pushed: {kernel_slug}")

    # Clean up the copied kernel-metadata.json
    if metadata_dst.exists() and metadata_dst != metadata_src:
        metadata_dst.unlink()

    print("\n  Check status:")
    print(f"    kaggle kernels status {kernel_slug}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Perch ensemble artifacts to Kaggle and push kernels"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/perch_ensemble_v1.yaml)",
    )
    parser.add_argument(
        "--push-kernel",
        action="store_true",
        help="Also push the submission kernel after uploading artifacts",
    )
    parser.add_argument(
        "--perch-only",
        action="store_true",
        help="Push Perch-only kernel (Phase A). Default is ensemble kernel (Phase B).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Check kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: kaggle CLI not found or not configured")
        sys.exit(1)

    # Resolve config
    experiment_dir = Path(__file__).resolve().parent.parent
    config_path = (
        Path(args.config) if args.config else experiment_dir / "config" / "perch_ensemble_v1.yaml"
    )

    config = load_config(config_path)
    paths = config["paths"]
    kaggle = config["kaggle"]

    probe_artifacts_dir = Path(paths["probe_artifacts_dir"])
    kaggle_user = kaggle["user"]
    artifacts_slug = kaggle["perch_artifacts_slug"]

    print("=" * 50)
    print("Experiment 018: Perch Ensemble — Kaggle Upload")
    print("=" * 50)
    print(f"  Config: {config_path}")
    print(f"  Artifacts: {probe_artifacts_dir}")
    print(f"  Kaggle user: {kaggle_user}")
    print(f"  Mode: {'Perch-only (Phase A)' if args.perch_only else 'Ensemble (Phase B)'}")

    # 1. Upload probe artifacts dataset
    upload_dataset(artifacts_slug, probe_artifacts_dir, kaggle_user, args.dry_run)

    # 2. Push kernel (if requested)
    if args.push_kernel:
        kernel_dir = experiment_dir / "kaggle_kernel"
        if args.perch_only:
            metadata_file = "kernel-metadata-perch-only.json"
        else:
            metadata_file = "kernel-metadata-ensemble-v1.json"
        push_kernel(kernel_dir, metadata_file, args.dry_run)

    print(f"\n{'=' * 50}")
    print("Done!")
    print("=" * 50)

    if not args.push_kernel:
        print("\n  To also push the kernel, re-run with --push-kernel")
        print("  Phase A (Perch-only): --push-kernel --perch-only")
        print("  Phase B (Ensemble):   --push-kernel")


if __name__ == "__main__":
    main()
