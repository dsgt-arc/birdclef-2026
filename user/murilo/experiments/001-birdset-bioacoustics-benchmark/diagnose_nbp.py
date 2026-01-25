#!/usr/bin/env python
"""Diagnostic script to investigate NBP inference issue.

Why is the model only processing 67 out of 100 files?
This script runs inference WITHOUT warning suppression to see what's happening.
"""

import sys
from pathlib import Path

# Add user package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import bioacoustics_model_zoo as bmz

from birdset_benchmark.datasets import load_birdset_dataset

print("=" * 80)
print("NBP Diagnostic - Finding why 67/100 files were processed")
print("=" * 80)
print()

# Load NBP dataset
print("Loading NBP dataset (100 samples)...")
hf_dataset = load_birdset_dataset(
    dataset_code="NBP",
    split="test_5s",
    subset_size=100,
)
print(f"✓ Loaded {len(hf_dataset)} samples")
print()

# Extract audio paths
print("Extracting audio paths...")
audio_paths = [hf_dataset[i]["audio"]["path"] for i in range(len(hf_dataset))]
print(f"✓ Extracted {len(audio_paths)} audio paths")
print()

# Check if files exist
print("Checking if audio files exist...")
missing = [p for p in audio_paths if not Path(p).exists()]
if missing:
    print(f"  [WARNING] {len(missing)} files don't exist:")
    for p in missing[:10]:
        print(f"    - {p}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
else:
    print(f"  ✓ All {len(audio_paths)} files exist")
print()

# Load model
print("Loading BirdSetConvNeXT model...")
model = bmz.BirdSetConvNeXT()
print("✓ Model loaded")
print()

# Run inference WITHOUT warning/error suppression
print("Running inference...")
print("=" * 80)
print("WARNINGS/ERRORS BELOW (if any):")
print("=" * 80)

pred_df = model.predict(
    audio_paths,
    batch_size=16,
    num_workers=4,
)

print("=" * 80)
print("END OF WARNINGS/ERRORS")
print("=" * 80)
print()

# Analyze results
print("Results:")
print(f"  Input files: {len(audio_paths)}")
print(f"  Output rows: {len(pred_df)}")
print(f"  Missing: {len(audio_paths) - len(pred_df)}")
print()

if len(pred_df) < len(audio_paths):
    print("Analyzing which files were skipped...")
    # pred_df has MultiIndex with 'file' as one level
    if hasattr(pred_df.index, "get_level_values"):
        pred_files = set(pred_df.index.get_level_values("file"))
        skipped = [p for p in audio_paths if p not in pred_files]

        print(f"\nSkipped {len(skipped)} files:")
        for i, path in enumerate(skipped[:20], 1):
            # Check if file exists and get size
            p = Path(path)
            if p.exists():
                size_mb = p.stat().st_size / 1024 / 1024
                print(f"  {i:2d}. {p.name} (Size: {size_mb:.2f} MB)")
            else:
                print(f"  {i:2d}. {p.name} (MISSING)")

        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

print()
print("=" * 80)
print("Diagnostic complete!")
print("=" * 80)
