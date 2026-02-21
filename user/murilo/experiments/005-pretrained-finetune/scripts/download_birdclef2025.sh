#!/bin/bash
# Download BirdCLEF 2025 competition data from Kaggle.
#
# Prerequisites:
#   1. Install kaggle CLI: pip install kaggle (or uv pip install kaggle)
#   2. Place API token at ~/.kaggle/kaggle.json
#      (Download from https://www.kaggle.com/settings → "Create New Token")
#   3. Accept competition rules at https://www.kaggle.com/competitions/birdclef-2025/rules
#
# Usage:
#   bash scripts/download_birdclef2025.sh
#   bash scripts/download_birdclef2025.sh --dry-run
#
# Data will be downloaded to:
#   ~/ps-clef2026_birdclef-0/data/birdclef-2025/
#
# Expected structure after download:
#   birdclef-2025/
#   ├── train_audio/           # Training audio files (.ogg) organized by species
#   │   ├── <species_code>/
#   │   │   ├── <xeno_canto_id>.ogg
#   │   │   └── ...
#   │   └── ...
#   ├── unlabeled_soundscapes/ # Unlabeled soundscape recordings
#   ├── test_soundscapes/      # Test soundscape recordings (empty until submission)
#   ├── train_metadata.csv     # Training metadata (species, location, quality, etc.)
#   ├── taxonomy.csv           # Species taxonomy information
#   └── sample_submission.csv  # Submission format template

set -eu

# ============================================================================
# Configuration
# ============================================================================

COMPETITION="birdclef-2025"
# Shared project storage (NOT home dir — home has 20GB quota)
DATA_ROOT="$HOME/ps-clef2026_birdclef-0/data"
DATA_DIR="$DATA_ROOT/$COMPETITION"

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Downloads BirdCLEF 2025 dataset to: $DATA_DIR"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be done without downloading"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Preflight checks
# ============================================================================

echo "=============================================="
echo "BirdCLEF 2025 Dataset Download"
echo "=============================================="
echo ""

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found."
    echo "Install with: uv pip install kaggle"
    echo "Then place API token at ~/.kaggle/kaggle.json"
    exit 1
fi
echo "  kaggle CLI: $(which kaggle)"

# Check API credentials
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "ERROR: Kaggle API token not found at ~/.kaggle/kaggle.json"
    echo "Download from: https://www.kaggle.com/settings → 'Create New Token'"
    exit 1
fi
echo "  API token: ~/.kaggle/kaggle.json"

# Show competition files
echo ""
echo "Competition files:"
kaggle competitions files "$COMPETITION"
echo ""

# ============================================================================
# Download
# ============================================================================

echo "Download target: $DATA_DIR"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] Would run:"
    echo "  mkdir -p $DATA_DIR"
    echo "  kaggle competitions download -c $COMPETITION -p $DATA_DIR"
    echo "  unzip $DATA_DIR/$COMPETITION.zip -d $DATA_DIR"
    echo "  rm $DATA_DIR/$COMPETITION.zip"
    exit 0
fi

# Create target directory
mkdir -p "$DATA_DIR"

# Download
echo "Downloading $COMPETITION dataset..."
echo "This may take a while (~30-50 GB)..."
echo ""
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

# Unzip
echo ""
echo "Extracting archive..."
ZIP_FILE="$DATA_DIR/$COMPETITION.zip"
if [[ -f "$ZIP_FILE" ]]; then
    unzip -o "$ZIP_FILE" -d "$DATA_DIR"
    echo ""
    echo "Removing zip archive to save space..."
    rm "$ZIP_FILE"
else
    echo "WARNING: Expected zip file not found at $ZIP_FILE"
    echo "Files in $DATA_DIR:"
    ls -la "$DATA_DIR"
fi

# ============================================================================
# Verification
# ============================================================================

echo ""
echo "=============================================="
echo "Download Complete"
echo "=============================================="
echo ""
echo "Contents of $DATA_DIR:"
ls -la "$DATA_DIR"
echo ""

# Count audio files
if [[ -d "$DATA_DIR/train_audio" ]]; then
    NUM_SPECIES=$(find "$DATA_DIR/train_audio" -mindepth 1 -maxdepth 1 -type d | wc -l)
    NUM_FILES=$(find "$DATA_DIR/train_audio" -name "*.ogg" | wc -l)
    echo "Training audio:"
    echo "  Species: $NUM_SPECIES"
    echo "  Audio files: $NUM_FILES"
fi

if [[ -f "$DATA_DIR/train_metadata.csv" ]]; then
    NUM_ROWS=$(($(wc -l < "$DATA_DIR/train_metadata.csv") - 1))
    echo "  Metadata rows: $NUM_ROWS"
fi

echo ""
echo "Total disk usage:"
du -sh "$DATA_DIR"
echo ""
echo "Data is ready at: $DATA_DIR"
