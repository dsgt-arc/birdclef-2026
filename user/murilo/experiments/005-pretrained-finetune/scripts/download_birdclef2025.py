#!/usr/bin/env python3
"""Download BirdCLEF 2025 competition data from Kaggle.

Prerequisites:
    1. pip install kaggle (or uv pip install kaggle)
    2. Authenticate via ONE of:
       a) export KAGGLE_API_TOKEN=KGAT_...  (newer token format)
       b) Place API token at ~/.kaggle/kaggle.json
    3. Accept competition rules at https://www.kaggle.com/competitions/birdclef-2025/rules

Usage:
    python scripts/download_birdclef2025.py
    python scripts/download_birdclef2025.py --dry-run

Data will be downloaded to:
    ~/ps-clef2026_birdclef-0/data/birdclef-2025/

Expected structure after download:
    birdclef-2025/
    ├── train_audio/           # Training audio (.ogg) organized by species
    │   ├── <species_code>/
    │   │   └── <xeno_canto_id>.ogg
    │   └── ...
    ├── unlabeled_soundscapes/ # Unlabeled soundscape recordings
    ├── test_soundscapes/      # Test soundscapes (empty until submission)
    ├── train_metadata.csv     # Training metadata
    ├── taxonomy.csv           # Species taxonomy
    └── sample_submission.csv  # Submission format
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "birdclef-2025"
DATA_ROOT = Path.home() / "ps-clef2026_birdclef-0" / "data"
DATA_DIR = DATA_ROOT / COMPETITION


def check_kaggle_cli() -> bool:
    """Check if kaggle CLI is available."""
    if shutil.which("kaggle") is None:
        print("ERROR: kaggle CLI not found.")
        print("Install with: uv pip install kaggle")
        return False
    print(f"  kaggle CLI: {shutil.which('kaggle')}")
    return True


def check_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("  Auth: KAGGLE_API_TOKEN env var")
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"  Auth: {kaggle_json}")
        return True
    print("ERROR: No Kaggle credentials found.")
    print("Set one of:")
    print("  export KAGGLE_API_TOKEN=KGAT_...")
    print("  OR place token at ~/.kaggle/kaggle.json")
    return False


def list_competition_files():
    """Show competition files."""
    print("\nCompetition files:")
    subprocess.run(["kaggle", "competitions", "files", COMPETITION], check=True)
    print()


def download(dry_run: bool = False):
    """Download and extract the dataset."""
    print(f"Download target: {DATA_DIR}\n")

    if dry_run:
        print("[DRY-RUN] Would run:")
        print(f"  mkdir -p {DATA_DIR}")
        print(f"  kaggle competitions download -c {COMPETITION} -p {DATA_DIR}")
        print(f"  unzip {DATA_DIR}/{COMPETITION}.zip -d {DATA_DIR}")
        print(f"  rm {DATA_DIR}/{COMPETITION}.zip")
        return

    # Create target directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"Downloading {COMPETITION} dataset...")
    print("This may take a while (~30-50 GB)...\n")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
        check=True,
    )

    # Extract
    zip_file = DATA_DIR / f"{COMPETITION}.zip"
    if zip_file.exists():
        print("\nExtracting archive...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(DATA_DIR)
        print("Removing zip archive to save space...")
        zip_file.unlink()
    else:
        print(f"WARNING: Expected zip file not found at {zip_file}")
        print(f"Files in {DATA_DIR}: {list(DATA_DIR.iterdir())}")


def verify():
    """Verify download integrity."""
    print("=" * 46)
    print("Download Complete")
    print("=" * 46)
    print(f"\nContents of {DATA_DIR}:")
    for p in sorted(DATA_DIR.iterdir()):
        kind = "dir" if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}")

    # Count audio files
    train_audio = DATA_DIR / "train_audio"
    if train_audio.exists():
        species_dirs = [d for d in train_audio.iterdir() if d.is_dir()]
        audio_files = list(train_audio.rglob("*.ogg"))
        print("\nTraining audio:")
        print(f"  Species: {len(species_dirs)}")
        print(f"  Audio files: {len(audio_files)}")

    # Count metadata rows
    metadata = DATA_DIR / "train_metadata.csv"
    if metadata.exists():
        num_rows = sum(1 for _ in metadata.open()) - 1  # subtract header
        print(f"  Metadata rows: {num_rows}")

    # Disk usage
    total_bytes = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024**3)
    print(f"\nTotal disk usage: {total_gb:.1f} GB")
    print(f"\nData is ready at: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download BirdCLEF 2025 dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    args = parser.parse_args()

    print("=" * 46)
    print("BirdCLEF 2025 Dataset Download")
    print("=" * 46)
    print()

    # Preflight checks
    if not check_kaggle_cli():
        sys.exit(1)
    if not check_credentials():
        sys.exit(1)

    list_competition_files()
    download(dry_run=args.dry_run)

    if not args.dry_run:
        verify()


if __name__ == "__main__":
    main()
