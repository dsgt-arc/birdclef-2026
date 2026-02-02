#!/usr/bin/env python3
"""Create and push a Kaggle kernel/notebook programmatically.

Usage:
    python scripts/create_kaggle_kernel.py [--push]
"""

import argparse
import json
import subprocess
from pathlib import Path

# Configuration
KAGGLE_USER = "cheungus"
COMPETITION = "birdclef-2025"
KERNEL_SLUG = "birdclef-2026-submission"
KERNEL_TITLE = "BirdCLEF 2026 Submission"

# Dataset slugs (must match upload_datasets.py)
DATASETS = [
    f"{KAGGLE_USER}/birdclef-2026-deps",
    f"{KAGGLE_USER}/birdclef-2026-code",
    f"{KAGGLE_USER}/birdclef-2026-weights",
]


def get_kernel_dir() -> Path:
    """Get the kernel directory path."""
    script_dir = Path(__file__).parent
    experiment_dir = script_dir.parent
    return experiment_dir / "kaggle_kernel"


def create_kernel_metadata(kernel_dir: Path) -> None:
    """Create kernel-metadata.json for Kaggle."""
    metadata = {
        "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "submission.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": DATASETS,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    
    meta_path = kernel_dir / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Created: {meta_path}")


def copy_inference_script(kernel_dir: Path) -> None:
    """Copy inference template to kernel directory."""
    script_dir = Path(__file__).parent
    experiment_dir = script_dir.parent
    template_path = experiment_dir / "templates" / "kaggle_inference.py"
    
    target_path = kernel_dir / "submission.py"
    target_path.write_text(template_path.read_text())
    print(f"Copied: {template_path} -> {target_path}")


def push_kernel(kernel_dir: Path, dry_run: bool = False) -> None:
    """Push kernel to Kaggle."""
    if dry_run:
        print(f"\n[DRY RUN] Would run: kaggle kernels push -p {kernel_dir}")
        return
    
    print(f"\nPushing kernel to Kaggle...")
    cmd = ["kaggle", "kernels", "push", "-p", str(kernel_dir)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"\n✓ Kernel pushed successfully!")
        print(f"  View at: https://www.kaggle.com/code/{KAGGLE_USER}/{KERNEL_SLUG}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and push Kaggle kernel")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push kernel to Kaggle after creating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without pushing",
    )
    args = parser.parse_args()
    
    print("Kaggle Kernel Creator")
    print("=" * 50)
    print(f"User: {KAGGLE_USER}")
    print(f"Competition: {COMPETITION}")
    print(f"Kernel: {KERNEL_SLUG}")
    print()
    
    # Create kernel directory
    kernel_dir = get_kernel_dir()
    kernel_dir.mkdir(parents=True, exist_ok=True)
    print(f"Kernel dir: {kernel_dir}")
    
    # Create metadata and copy script
    create_kernel_metadata(kernel_dir)
    copy_inference_script(kernel_dir)
    
    print(f"\n✓ Kernel files created in: {kernel_dir}")
    print(f"  - kernel-metadata.json")
    print(f"  - submission.py")
    
    # Push to Kaggle if requested
    if args.push or args.dry_run:
        push_kernel(kernel_dir, dry_run=args.dry_run)
    else:
        print(f"\nTo push to Kaggle, run:")
        print(f"  python {Path(__file__).name} --push")
        print(f"\nOr manually:")
        print(f"  kaggle kernels push -p {kernel_dir}")


if __name__ == "__main__":
    main()
