"""BamBird runner: extract temporal event boundaries (ROIs) from audio.

Uses scikit-maad for spectrogram computation and ROI extraction.
BamBird identifies acoustic events without species classification —
it produces temporal boundaries that can be used as one labeling function
in the weak supervision pipeline.

Output: parquet with columns [file, events_json, num_rois, duration_s]
  - events_json: JSON string with list of {min_t, max_t, min_f, max_f, duration_s}
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import typer
from maad import sound, rois
from maad.util import power2dB

from teacher_labels.utils import (
    list_audio_files,
    load_config,
    partition_files,
    ensure_output_dir,
)

app = typer.Typer()


def extract_rois(
    audio_path: Path,
    min_duration: float = 0.05,
    max_duration: float = 10.0,
) -> list[dict]:
    """Extract acoustic ROIs from a single audio file using scikit-maad.

    Args:
        audio_path: Path to audio file.
        min_duration: Minimum ROI duration in seconds.
        max_duration: Maximum ROI duration in seconds.

    Returns:
        List of ROI dicts with min_t, max_t, min_f, max_f, duration_s.
    """
    s, sr = sf.read(str(audio_path))
    # Convert to mono if stereo
    if s.ndim > 1:
        s = s.mean(axis=1)
    # Compute spectrogram
    Sxx, tn, fn, ext = sound.spectrogram(s, sr, nperseg=1024, noverlap=512)
    # Convert to dB
    Sxx_db = power2dB(Sxx)
    # Find ROIs using the binary method
    # create_mask can fail on silent/unusual audio (empty percentile array)
    try:
        Sxx_bin = rois.create_mask(Sxx_db, mode_bin="relative", bin_h=6, bin_l=1)
        rois_df = rois.select_rois(Sxx_bin, tn, fn)
    except (IndexError, ValueError):
        return []

    if rois_df.empty:
        return []

    events = []
    for _, roi in rois_df.iterrows():
        duration = roi["max_t"] - roi["min_t"]
        if duration < min_duration or duration > max_duration:
            continue
        events.append({
            "min_t": round(float(roi["min_t"]), 4),
            "max_t": round(float(roi["max_t"]), 4),
            "min_f": round(float(roi["min_f"]), 1),
            "max_f": round(float(roi["max_f"]), 1),
            "duration_s": round(duration, 4),
        })
    return events


def process_partition(
    files: list[Path],
    output_path: Path,
    min_duration: float = 0.05,
    max_duration: float = 10.0,
) -> pd.DataFrame:
    """Process a partition of audio files and save results.

    Args:
        files: List of audio file paths.
        output_path: Path to save the output parquet.
        min_duration: Minimum ROI duration.
        max_duration: Maximum ROI duration.

    Returns:
        DataFrame with results.
    """
    records = []
    for i, fpath in enumerate(files):
        try:
            info = sf.info(str(fpath))
            duration_s = info.duration
            events = extract_rois(fpath, min_duration, max_duration)
            records.append({
                "file": str(fpath),
                "events_json": json.dumps(events),
                "num_rois": len(events),
                "duration_s": round(duration_s, 2),
            })
        except Exception:
            traceback.print_exc()
            records.append({
                "file": str(fpath),
                "events_json": "[]",
                "num_rois": 0,
                "duration_s": 0.0,
            })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(files)} files")

    df = pd.DataFrame(records)
    ensure_output_dir(output_path.parent)
    df.to_parquet(output_path, index=False)
    return df


@app.command()
def run(
    config: str = "configs/teacher.yaml",
    source: str = typer.Argument(..., help="'focal' or 'soundscape'"),
    partition_id: int = typer.Argument(..., help="Partition ID (0-indexed)"),
):
    """Run BamBird ROI extraction on a partition of audio files."""
    cfg = load_config(config)
    source_key = "focal_dir" if source == "focal" else "soundscape_dir"
    audio_dir = cfg["data"][source_key]
    num_partitions = cfg["partitions"][source]
    bambird_cfg = cfg["models"]["bambird"]

    source_label = "train_audio" if source == "focal" else "train_soundscapes"
    output_dir = Path(cfg["output"]["root"]).expanduser() / "bambird" / source_label
    output_path = output_dir / f"part_{partition_id:04d}.parquet"

    if output_path.exists():
        print(f"Output already exists: {output_path}, skipping.")
        return

    files = list_audio_files(audio_dir, source)
    partition = partition_files(files, num_partitions, partition_id)

    print(f"BamBird {source} partition {partition_id}/{num_partitions}")
    print(f"  Total files: {len(files)}, this partition: {len(partition)}")

    start = time.time()
    df = process_partition(
        partition,
        output_path,
        min_duration=bambird_cfg["min_roi_duration"],
        max_duration=bambird_cfg["max_roi_duration"],
    )
    elapsed = time.time() - start

    total_rois = df["num_rois"].sum()
    print(f"  Done in {elapsed:.1f}s — {len(partition)} files, {total_rois} ROIs")
    print(f"  Output: {output_path}")


def main():
    app()


if __name__ == "__main__":
    main()
