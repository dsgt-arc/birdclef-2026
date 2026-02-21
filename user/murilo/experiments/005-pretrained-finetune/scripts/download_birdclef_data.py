#!/usr/bin/env python3
"""Download BirdCLEF competition data from Kaggle.

Supports any BirdCLEF competition year (2025, 2026, etc.).

Prerequisites:
    1. pip install kaggle (or uv pip install kaggle)
    2. Authenticate via ONE of:
       a) Add KAGGLE_API_TOKEN=KGAT_... to user/murilo/.env
       b) export KAGGLE_API_TOKEN=KGAT_...  (session only)
       c) Place API token at ~/.kaggle/kaggle.json
    3. Accept competition rules at https://www.kaggle.com/competitions/birdclef-2025/rules

Usage:
    python scripts/download_birdclef_data.py                          # default: birdclef-2025
    python scripts/download_birdclef_data.py --competition birdclef-2026
    python scripts/download_birdclef_data.py --dry-run

Data will be downloaded to:
    ~/ps-clef2026_birdclef-0/data/<competition>/
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env from user/murilo/ (contains KAGGLE_API_TOKEN, WANDB_API_KEY, etc.)
# scripts/ → 005-pretrained-finetune/ → experiments/ → murilo/
_user_dir = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_user_dir / ".env")

DEFAULT_COMPETITION = "birdclef-2025"
DATA_ROOT = Path.home() / "ps-clef2026_birdclef-0" / "data"


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


def list_competition_files(competition: str):
    """Show competition files."""
    print("\nCompetition files:")
    subprocess.run(["kaggle", "competitions", "files", competition], check=True)
    print()


def download(competition: str, data_dir: Path, dry_run: bool = False):
    """Download and extract the dataset."""
    print(f"Download target: {data_dir}\n")

    if dry_run:
        print("[DRY-RUN] Would run:")
        print(f"  mkdir -p {data_dir}")
        print(f"  kaggle competitions download -c {competition} -p {data_dir}")
        print(f"  unzip {data_dir}/{competition}.zip -d {data_dir}")
        print(f"  rm {data_dir}/{competition}.zip")
        return

    # Create target directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"Downloading {competition} dataset...")
    print("This may take a while depending on dataset size...\n")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(data_dir)],
        check=True,
    )

    # Extract
    zip_file = data_dir / f"{competition}.zip"
    if zip_file.exists():
        print("\nExtracting archive...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(data_dir)
        print("Removing zip archive to save space...")
        zip_file.unlink()
    else:
        print(f"WARNING: Expected zip file not found at {zip_file}")
        print(f"Files in {data_dir}: {list(data_dir.iterdir())}")


def verify(data_dir: Path):
    """Verify download integrity."""
    print("=" * 46)
    print("Download Complete")
    print("=" * 46)
    print(f"\nContents of {data_dir}:")
    for p in sorted(data_dir.iterdir()):
        kind = "dir" if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}")

    # Count audio files
    train_audio = data_dir / "train_audio"
    if train_audio.exists():
        species_dirs = [d for d in train_audio.iterdir() if d.is_dir()]
        audio_files = list(train_audio.rglob("*.ogg"))
        print("\nTraining audio:")
        print(f"  Species: {len(species_dirs)}")
        print(f"  Audio files: {len(audio_files)}")

    # Count metadata rows
    for csv_name in ("train.csv", "train_metadata.csv"):
        metadata = data_dir / csv_name
        if metadata.exists():
            num_rows = sum(1 for _ in metadata.open()) - 1  # subtract header
            print(f"  Metadata rows ({csv_name}): {num_rows}")
            break

    # Disk usage
    total_bytes = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024**3)
    print(f"\nTotal disk usage: {total_gb:.1f} GB")
    print(f"\nData is ready at: {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download BirdCLEF competition data")
    parser.add_argument(
        "--competition",
        "-c",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    args = parser.parse_args()

    competition = args.competition
    data_dir = DATA_ROOT / competition

    print("=" * 46)
    print(f"BirdCLEF Dataset Download: {competition}")
    print("=" * 46)
    print()

    # Preflight checks
    if not check_kaggle_cli():
        sys.exit(1)
    if not check_credentials():
        sys.exit(1)

    list_competition_files(competition)
    download(competition, data_dir, dry_run=args.dry_run)

    if not args.dry_run:
        verify(data_dir)


if __name__ == "__main__":
    main()
