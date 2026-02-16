"""Compute statistics from teacher label outputs.

Reads all parquet outputs and computes per-model/per-source summaries:
- Species frequency distributions
- Confidence score distributions
- BamBird event density and duration stats
- Model agreement (BirdNET vs Perch top species)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import typer

from teacher_labels.utils import load_config, ensure_output_dir

app = typer.Typer()


def bambird_stats(label_dir: Path) -> dict:
    """Compute stats from BamBird outputs."""
    stats = {}
    for source_label in ["train_audio", "train_soundscapes"]:
        source_dir = label_dir / "bambird" / source_label
        if not source_dir.exists():
            continue
        parts = sorted(source_dir.glob("part_*.parquet"))
        if not parts:
            continue

        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        events_per_file = df["num_rois"]
        durations = df["duration_s"]

        # Parse events for duration stats
        all_events = []
        for events_json in df["events_json"]:
            events = json.loads(events_json)
            all_events.extend(events)

        event_durations = [e["duration_s"] for e in all_events]

        stats[source_label] = {
            "num_files": len(df),
            "total_rois": int(events_per_file.sum()),
            "rois_per_file_mean": round(float(events_per_file.mean()), 2),
            "rois_per_file_median": round(float(events_per_file.median()), 2),
            "rois_per_file_max": int(events_per_file.max()),
            "files_with_no_rois": int((events_per_file == 0).sum()),
            "audio_duration_total_h": round(float(durations.sum()) / 3600, 2),
            "event_duration_mean_s": round(float(np.mean(event_durations)), 4) if event_durations else 0,
            "event_duration_median_s": round(float(np.median(event_durations)), 4) if event_durations else 0,
        }

    return stats


def embedding_model_stats(label_dir: Path, model_name: str) -> dict:
    """Compute stats from BirdNET or Perch outputs.

    Uses polars-native operations to avoid loading all species columns into
    numpy (which would OOM for BirdNET's 6522 species x ~3M rows).
    """
    stats = {}
    for source_label in ["train_audio", "train_soundscapes"]:
        predict_dir = label_dir / model_name / source_label / "parts" / "predict"
        embed_dir = label_dir / model_name / source_label / "parts" / "embed"

        if not predict_dir.exists():
            continue
        predict_parts = sorted(predict_dir.glob("part_*.parquet"))
        if not predict_parts:
            continue

        # Use polars lazy scan — only materialize what we need
        lf = pl.scan_parquet(predict_dir / "*.parquet")

        # Get species columns (exclude metadata) from schema
        meta_cols = {"file", "start_time", "end_time", "index"}
        all_cols = lf.collect_schema().names()
        species_cols = [c for c in all_cols if c not in meta_cols]

        # Top species by mean confidence (polars-native, no numpy)
        mean_df = lf.select(species_cols).mean().collect()
        mean_vals = {col: float(mean_df[col][0]) for col in species_cols}
        top_species = sorted(mean_vals.items(), key=lambda x: -x[1])[:20]

        # Row count
        num_clips = lf.select(pl.len()).collect().item()

        # Max confidence per row using polars max_horizontal (no numpy copy)
        max_conf_lf = lf.select(pl.max_horizontal(species_cols).alias("max_conf"))
        max_conf_series = max_conf_lf.collect()["max_conf"]

        max_conf_mean = float(max_conf_series.mean())
        max_conf_median = float(max_conf_series.median())
        max_conf_p95 = float(max_conf_series.quantile(0.95))

        # Embedding stats (just read schema from first partition)
        embed_shape = None
        embed_size_mb = 0
        if embed_dir.exists():
            embed_parts = sorted(embed_dir.glob("part_*.parquet"))
            if embed_parts:
                sample_schema = pl.scan_parquet(embed_parts[0]).collect_schema()
                embed_dim = len(sample_schema) - len(meta_cols.intersection(set(sample_schema.names())))
                embed_shape = [num_clips, embed_dim]
                embed_size_mb = sum(p.stat().st_size for p in embed_parts) / 1e6

        predict_size_mb = sum(p.stat().st_size for p in predict_parts) / 1e6

        # Timing
        timing_dir = label_dir / model_name / source_label / "parts" / "timing"
        total_time_s = 0
        if timing_dir.exists():
            for tp in timing_dir.glob("*.json"):
                with open(tp) as f:
                    total_time_s += json.load(f).get("elapsed_s", 0)

        stats[source_label] = {
            "num_clips": num_clips,
            "num_species": len(species_cols),
            "num_partitions": len(predict_parts),
            "top_20_species": [{"species": s, "mean_conf": round(c, 4)} for s, c in top_species],
            "max_conf_mean": round(max_conf_mean, 4),
            "max_conf_median": round(max_conf_median, 4),
            "max_conf_p95": round(max_conf_p95, 4),
            "predict_size_mb": round(predict_size_mb, 1),
            "embed_shape": embed_shape,
            "embed_size_mb": round(embed_size_mb, 1),
            "total_time_s": round(total_time_s, 1),
        }

    return stats


def model_agreement(label_dir: Path) -> dict:
    """Compute agreement between BirdNET and Perch top predictions.

    Uses lazy scans to only read schemas and counts — no full data load.
    """
    agreement = {}
    for source_label in ["train_audio", "train_soundscapes"]:
        bn_dir = label_dir / "birdnet" / source_label / "parts" / "predict"
        perch_dir = label_dir / "perch" / source_label / "parts" / "predict"

        if not bn_dir.exists() or not perch_dir.exists():
            continue

        bn_lf = pl.scan_parquet(bn_dir / "*.parquet")
        perch_lf = pl.scan_parquet(perch_dir / "*.parquet")

        # Get species columns from schema (no data load)
        meta_cols = {"file", "start_time", "end_time", "index"}
        bn_all = bn_lf.collect_schema().names()
        perch_all = perch_lf.collect_schema().names()
        bn_species = [c for c in bn_all if c not in meta_cols]
        perch_species = [c for c in perch_all if c not in meta_cols]
        common_species = set(bn_species) & set(perch_species)

        if not common_species:
            agreement[source_label] = {"common_species": 0, "note": "no overlapping species"}
            continue

        # Clip counts via lazy aggregation
        bn_clips = bn_lf.select(pl.len()).collect().item()
        perch_clips = perch_lf.select(pl.len()).collect().item()

        agreement[source_label] = {
            "birdnet_species": len(bn_species),
            "perch_species": len(perch_species),
            "common_species": len(common_species),
            "birdnet_clips": bn_clips,
            "perch_clips": perch_clips,
        }

    return agreement


@app.command()
def run(
    config: str = "configs/teacher.yaml",
    output: str = typer.Option(None, help="Path to save stats JSON"),
):
    """Compute summary statistics from all teacher label outputs."""
    cfg = load_config(config)
    label_dir = Path(cfg["output"]["root"]).expanduser()

    print("Computing BamBird stats...")
    bambird = bambird_stats(label_dir)

    print("Computing BirdNET stats...")
    birdnet = embedding_model_stats(label_dir, "birdnet")

    print("Computing Perch stats...")
    perch = embedding_model_stats(label_dir, "perch")

    print("Computing model agreement...")
    agree = model_agreement(label_dir)

    results = {
        "bambird": bambird,
        "birdnet": birdnet,
        "perch": perch,
        "model_agreement": agree,
    }

    # Total storage
    total_mb = 0
    for model_stats in [birdnet, perch]:
        for src in model_stats.values():
            total_mb += src.get("predict_size_mb", 0)
            total_mb += src.get("embed_size_mb", 0)
    results["total_storage_mb"] = round(total_mb, 1)

    out_str = json.dumps(results, indent=2)
    print(f"\n{out_str}")

    if output:
        out_path = Path(output).expanduser()
        ensure_output_dir(out_path.parent)
        with open(out_path, "w") as f:
            f.write(out_str)
        print(f"\nSaved to {out_path}")


def main():
    app()


if __name__ == "__main__":
    main()
