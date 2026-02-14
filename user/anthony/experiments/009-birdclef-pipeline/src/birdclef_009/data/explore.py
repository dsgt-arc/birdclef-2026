"""Data exploration utilities for BirdCLEF datasets.

Computes comprehensive statistics: total durations, per-species distributions,
file format details, and pretraining corpus size.

Outputs a joinable CSV with per-file metadata (filename, duration, sample_rate)
that can be merged with the train metadata CSV.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)


def _probe_file(path_str):
    """Probe a single audio file for duration and sample rate.

    Returns (relative_path, duration_s, sample_rate) or None on error.
    """
    try:
        info = sf.info(path_str)
        return (path_str, info.duration, info.samplerate, info.frames)
    except Exception:
        return None


def _scan_audio_durations(audio_dir, extensions=None, max_workers=16):
    """Scan all audio files in a directory in parallel.

    Args:
        audio_dir: Path to directory
        extensions: Set of extensions to include
        max_workers: Thread pool size

    Returns:
        List of (relative_path, duration_s, sample_rate, frames) tuples
    """
    if extensions is None:
        extensions = {".ogg", ".wav", ".flac", ".mp3"}

    audio_dir = Path(audio_dir)
    files = sorted(f for f in audio_dir.rglob("*") if f.suffix.lower() in extensions)
    logger.info(f"  Found {len(files)} audio files, scanning with {max_workers} threads...")

    results = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_probe_file, str(f)): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                # Store relative path
                abs_path = result[0]
                rel_path = str(Path(abs_path).relative_to(audio_dir))
                results.append((rel_path, result[1], result[2], result[3]))
            else:
                errors += 1
            if (i + 1) % 5000 == 0:
                logger.info(f"  ... scanned {i + 1}/{len(files)} files")

    if errors:
        logger.warning(f"  Failed to read {errors}/{len(files)} files")

    return results


def _format_duration(seconds):
    """Format seconds as human-readable string."""
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:.1f}h ({seconds:.0f}s)"
    minutes = seconds / 60
    return f"{minutes:.1f}min ({seconds:.0f}s)"


def explore_dataset(data_dir: str, output_dir: str | None = None, max_workers: int = 16) -> dict:
    """Comprehensive EDA of a BirdCLEF dataset directory.

    Outputs:
    - Summary statistics to logger and JSON
    - Per-file CSV: filename, duration_s, sample_rate, frames (joinable with train.csv)
    - Per-species CSV: species, num_recordings, total_duration_s, mean_duration_s

    Args:
        data_dir: Path to BirdCLEF dataset root
        output_dir: Where to write CSV files (default: data_dir itself)
        max_workers: Thread pool size for parallel file scanning

    Returns:
        Dict with all exploration results
    """
    data_path = Path(data_dir).expanduser()
    out_path = Path(output_dir).expanduser() if output_dir else data_path
    results = {"path": str(data_path), "exists": data_path.exists()}

    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        return results

    # Top-level contents
    contents = sorted([p.name for p in data_path.iterdir()])
    results["contents"] = contents
    logger.info(f"Dataset root: {data_path}")
    logger.info(f"Contents: {contents}")
    logger.info("")

    # ── Metadata CSV ──
    metadata_path = None
    for name in ["train.csv", "train_metadata.csv"]:
        candidate = data_path / name
        if candidate.exists():
            metadata_path = candidate
            break

    metadata_df = None
    if metadata_path is not None:
        metadata_df = pd.read_csv(metadata_path)
        species_col = "primary_label" if "primary_label" in metadata_df.columns else None

        meta_info = {
            "filename": metadata_path.name,
            "rows": len(metadata_df),
            "columns": list(metadata_df.columns),
        }

        if species_col:
            species_counts = metadata_df[species_col].value_counts()
            meta_info["species_count"] = int(species_counts.shape[0])
            meta_info["species_list"] = sorted(metadata_df[species_col].unique().tolist())
            meta_info["recordings_per_species"] = {
                "min": int(species_counts.min()),
                "max": int(species_counts.max()),
                "mean": float(species_counts.mean()),
                "median": float(species_counts.median()),
            }
            meta_info["most_recordings"] = species_counts.head(10).to_dict()
            meta_info["fewest_recordings"] = species_counts.tail(10).to_dict()

        results["metadata"] = meta_info

        logger.info(f"── Metadata: {metadata_path.name} ──")
        logger.info(f"  Rows: {len(metadata_df)}")
        logger.info(f"  Columns: {list(metadata_df.columns)}")
        if species_col:
            logger.info(f"  Species: {meta_info['species_count']}")
            logger.info(f"  Recordings/species: min={species_counts.min()}, "
                        f"max={species_counts.max()}, median={species_counts.median():.0f}, "
                        f"mean={species_counts.mean():.1f}")
            logger.info(f"  Most recordings: {dict(species_counts.head(5))}")
            logger.info(f"  Fewest recordings: {dict(species_counts.tail(5))}")
        logger.info("")

    # ── Train audio ──
    train_audio = data_path / "train_audio"
    if train_audio.exists():
        logger.info("── Train Audio ──")
        file_info = _scan_audio_durations(train_audio, max_workers=max_workers)

        durations = np.array([d for _, d, _, _ in file_info])
        sample_rates = set(sr for _, _, sr, _ in file_info)

        # Write per-file CSV (joinable with train.csv on "filename")
        file_df = pd.DataFrame(file_info, columns=["filename", "duration_s", "sample_rate", "frames"])
        file_csv_path = out_path / "train_audio_durations.csv"
        file_df.to_csv(file_csv_path, index=False)
        logger.info(f"  Wrote per-file CSV: {file_csv_path}")

        train_stats = {
            "file_count": len(file_info),
            "total_duration_s": float(durations.sum()),
            "total_duration_h": float(durations.sum() / 3600),
            "extensions": list(set(Path(f).suffix for f, _, _, _ in file_info)),
            "sample_rates": sorted(sample_rates),
            "duration_stats": {
                "min": float(durations.min()),
                "max": float(durations.max()),
                "mean": float(durations.mean()),
                "median": float(np.median(durations)),
                "std": float(durations.std()),
                "p5": float(np.percentile(durations, 5)),
                "p25": float(np.percentile(durations, 25)),
                "p75": float(np.percentile(durations, 75)),
                "p95": float(np.percentile(durations, 95)),
            },
        }

        # Duration buckets
        buckets = [0, 5, 10, 30, 60, 120, 300, 600, float("inf")]
        bucket_labels = ["0-5s", "5-10s", "10-30s", "30-60s", "1-2min", "2-5min", "5-10min", "10min+"]
        bucket_counts = {}
        for i in range(len(buckets) - 1):
            count = int(((durations >= buckets[i]) & (durations < buckets[i + 1])).sum())
            bucket_counts[bucket_labels[i]] = count
        train_stats["duration_buckets"] = bucket_counts

        # Per-species duration (join file_df with metadata)
        if metadata_df is not None and "primary_label" in metadata_df.columns and "filename" in metadata_df.columns:
            merged = metadata_df.merge(file_df, on="filename", how="left")
            merged["duration_s"] = merged["duration_s"].fillna(0)

            species_agg = merged.groupby("primary_label").agg(
                num_recordings=("filename", "count"),
                total_duration_s=("duration_s", "sum"),
                mean_duration_s=("duration_s", "mean"),
                min_duration_s=("duration_s", "min"),
                max_duration_s=("duration_s", "max"),
                median_duration_s=("duration_s", "median"),
            ).sort_values("total_duration_s", ascending=False).reset_index()

            # Write per-species CSV
            species_csv_path = out_path / "species_duration_stats.csv"
            species_agg.to_csv(species_csv_path, index=False)
            logger.info(f"  Wrote per-species CSV: {species_csv_path}")

            train_stats["species_duration_stats"] = {
                "min_total_s": float(species_agg["total_duration_s"].min()),
                "max_total_s": float(species_agg["total_duration_s"].max()),
                "mean_total_s": float(species_agg["total_duration_s"].mean()),
                "median_total_s": float(species_agg["total_duration_s"].median()),
                "total_species": len(species_agg),
            }
            train_stats["most_audio_species"] = dict(
                zip(species_agg.head(10)["primary_label"], species_agg.head(10)["total_duration_s"])
            )
            train_stats["least_audio_species"] = dict(
                zip(species_agg.tail(10)["primary_label"], species_agg.tail(10)["total_duration_s"])
            )

            logger.info(f"  Per-species total duration:")
            logger.info(f"    min={_format_duration(species_agg['total_duration_s'].min())}, "
                        f"max={_format_duration(species_agg['total_duration_s'].max())}, "
                        f"median={_format_duration(species_agg['total_duration_s'].median())}")
            logger.info(f"    Most audio:")
            for _, row in species_agg.head(5).iterrows():
                logger.info(f"      {row['primary_label']}: {_format_duration(row['total_duration_s'])} "
                            f"({row['num_recordings']} recordings)")
            logger.info(f"    Least audio:")
            for _, row in species_agg.tail(5).iterrows():
                logger.info(f"      {row['primary_label']}: {_format_duration(row['total_duration_s'])} "
                            f"({row['num_recordings']} recordings)")

            # Also write the merged per-recording CSV (metadata + duration)
            merged_csv_path = out_path / "train_with_durations.csv"
            merged.to_csv(merged_csv_path, index=False)
            logger.info(f"  Wrote merged CSV: {merged_csv_path}")

        # 10s chunks estimate
        num_10s_chunks = int(durations.sum() / 10)
        train_stats["estimated_10s_chunks"] = num_10s_chunks

        results["train_audio"] = train_stats

        logger.info(f"  Files: {len(file_info)}")
        logger.info(f"  Total duration: {_format_duration(durations.sum())}")
        logger.info(f"  Duration: min={durations.min():.1f}s, max={durations.max():.1f}s, "
                    f"mean={durations.mean():.1f}s, median={np.median(durations):.1f}s")
        logger.info(f"  Sample rates: {sorted(sample_rates)}")
        logger.info(f"  Duration buckets: {bucket_counts}")
        logger.info(f"  Estimated 10s chunks: {num_10s_chunks:,}")
        logger.info("")

    # ── Soundscapes ──
    for soundscape_name in ["train_soundscapes", "test_soundscapes", "unlabeled_soundscapes"]:
        soundscape_dir = data_path / soundscape_name
        if soundscape_dir.exists():
            logger.info(f"── {soundscape_name} ──")
            file_info = _scan_audio_durations(soundscape_dir, max_workers=max_workers)

            if file_info:
                durations = np.array([d for _, d, _, _ in file_info])
                sample_rates = set(sr for _, _, sr, _ in file_info)

                # Write per-file CSV for soundscapes
                sc_df = pd.DataFrame(file_info, columns=["filename", "duration_s", "sample_rate", "frames"])
                sc_csv_path = out_path / f"{soundscape_name}_durations.csv"
                sc_df.to_csv(sc_csv_path, index=False)
                logger.info(f"  Wrote CSV: {sc_csv_path}")

                sc_stats = {
                    "file_count": len(file_info),
                    "total_duration_s": float(durations.sum()),
                    "total_duration_h": float(durations.sum() / 3600),
                    "sample_rates": sorted(sample_rates),
                    "duration_stats": {
                        "min": float(durations.min()),
                        "max": float(durations.max()),
                        "mean": float(durations.mean()),
                        "median": float(np.median(durations)),
                    },
                    "estimated_10s_chunks": int(durations.sum() / 10),
                }
                results[soundscape_name] = sc_stats

                logger.info(f"  Files: {len(file_info)}")
                logger.info(f"  Total duration: {_format_duration(durations.sum())}")
                logger.info(f"  Duration: min={durations.min():.1f}s, max={durations.max():.1f}s, "
                            f"mean={durations.mean():.1f}s")
                logger.info(f"  Sample rates: {sorted(sample_rates)}")
                logger.info(f"  Estimated 10s chunks: {sc_stats['estimated_10s_chunks']:,}")
            else:
                results[soundscape_name] = {"file_count": 0}
                logger.info(f"  No audio files found")
            logger.info("")

    # ── Sample submission ──
    submission_path = data_path / "sample_submission.csv"
    if submission_path.exists():
        sub_df = pd.read_csv(submission_path)
        species_cols = [c for c in sub_df.columns if c != "row_id"]
        results["sample_submission"] = {
            "rows": len(sub_df),
            "species_count": len(species_cols),
            "columns_preview": list(sub_df.columns[:10]),
            "example_row_ids": sub_df["row_id"].head(5).tolist() if "row_id" in sub_df.columns else [],
        }
        logger.info("── Sample Submission ──")
        logger.info(f"  Rows: {len(sub_df)}, Species columns: {len(species_cols)}")
        if "row_id" in sub_df.columns:
            logger.info(f"  Example row_ids: {sub_df['row_id'].head(5).tolist()}")
        logger.info("")

    # ── Taxonomy ──
    taxonomy_path = data_path / "taxonomy.csv"
    if taxonomy_path.exists():
        tax_df = pd.read_csv(taxonomy_path)
        results["taxonomy"] = {
            "rows": len(tax_df),
            "columns": list(tax_df.columns),
        }
        logger.info("── Taxonomy ──")
        logger.info(f"  Rows: {len(tax_df)}, Columns: {list(tax_df.columns)}")
        logger.info("")

    # ── Grand summary ──
    total_audio_h = 0
    total_10s_chunks = 0
    for key in ["train_audio", "train_soundscapes", "test_soundscapes", "unlabeled_soundscapes"]:
        if key in results and isinstance(results[key], dict):
            total_audio_h += results[key].get("total_duration_h", 0)
            total_10s_chunks += results[key].get("estimated_10s_chunks", 0)

    results["grand_total"] = {
        "total_audio_hours": total_audio_h,
        "total_10s_chunks": total_10s_chunks,
    }
    logger.info("── Grand Total ──")
    logger.info(f"  Total audio: {total_audio_h:.1f} hours")
    logger.info(f"  Total 10s chunks: {total_10s_chunks:,}")

    return results
