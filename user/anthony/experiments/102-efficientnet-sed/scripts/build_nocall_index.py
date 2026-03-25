#!/usr/bin/env python3
"""Build no-call chunk index from train soundscapes.

Strategy: For labeled soundscape files, enumerate all 5s windows. Windows that
appear in the labels CSV are "has-call", windows that don't are "no-call".
Only use labeled files (where we trust that unlabeled windows are truly silent).

Output: parquet with columns matching focal_activity.parquet format:
  filename, chunk_start, activity_score (set to 0.0 for no-call)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build no-call chunk index")
    parser.add_argument(
        "--soundscape-dir",
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes",
    )
    parser.add_argument(
        "--labels-csv",
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes_labels.csv",
    )
    parser.add_argument(
        "--train-csv",
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train.csv",
    )
    parser.add_argument("--output", default="artifacts/nocall_index.parquet")
    parser.add_argument("--chunk-duration", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    logger.info(f"Loaded {len(labels_df)} labeled windows from {args.labels_csv}")

    # Parse time strings to seconds
    def time_to_seconds(t):
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    labels_df["start_sec"] = labels_df["start"].apply(time_to_seconds)

    # Get species list from train.csv (for mapping IDs)
    train_df = pd.read_csv(args.train_csv)
    species_ids = set(train_df["primary_label"].astype(str).unique())
    logger.info(f"Focal species: {len(species_ids)}")

    # Build set of (filename, start_sec) that have labels
    labeled_windows = set()
    for _, row in labels_df.iterrows():
        labeled_windows.add((row["filename"], row["start_sec"]))

    # Get unique labeled filenames
    labeled_files = labels_df["filename"].unique()
    logger.info(f"Labeled soundscape files: {len(labeled_files)}")

    # For each labeled file, enumerate all 5s windows and find no-call ones
    soundscape_dir = Path(args.soundscape_dir)
    nocall_chunks = []
    has_call_count = 0

    for fname in sorted(labeled_files):
        fpath = soundscape_dir / fname
        if not fpath.exists():
            logger.warning(f"File not found: {fpath}")
            continue

        info = sf.info(str(fpath))
        duration = info.frames / info.samplerate

        # Enumerate 5s windows
        t = 0.0
        while t + args.chunk_duration <= duration + 0.1:
            if (fname, int(t)) in labeled_windows:
                has_call_count += 1
            else:
                nocall_chunks.append({
                    "filename": fname,
                    "chunk_start": t,
                    "activity_score": 0.0,
                })
            t += args.chunk_duration

    logger.info(f"Has-call windows: {has_call_count}")
    logger.info(f"No-call windows: {len(nocall_chunks)}")

    # Save
    df = pd.DataFrame(nocall_chunks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Summary
    total = has_call_count + len(nocall_chunks)
    logger.info(f"Total windows in labeled files: {total}")
    logger.info(f"No-call fraction: {len(nocall_chunks)/total:.1%}")


if __name__ == "__main__":
    main()
