"""BamBird runner: full 3-stage pipeline using the bambird library.

Stage 1 (extract-features): Per-file ROI extraction + wavelet feature computation
  - Uses bambird.single_file_extract_rois for bandpass + ROI detection
  - Uses bambird.compute_features for wavelet shape features per ROI
  - Saves features to parquet partitions
  - Designed to run as SLURM array job (one partition per task)

Stage 2 (cluster): Per-species DBSCAN clustering across all features
  - Uses bambird.find_cluster to separate signal from noise
  - Produces final labels: detected_events, event_cluster, auto_label per file
  - Single SLURM job after all extract jobs complete

Output: bambird_labels.parquet with columns:
  [file, species, detected_events, event_cluster, num_signal_rois, num_noise_rois]
"""

import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd
import typer

from teacher_labels.utils import (
    list_audio_files,
    load_config,
    partition_files,
    ensure_output_dir,
)

app = typer.Typer()


def _load_bambird_config(config_path: str) -> dict:
    """Load bambird-format YAML config and return the PARAMS dict."""
    from bambird.config import load_config
    return load_config(config_path)


def _extract_features_single_file(
    audio_path: str,
    params_extract: dict,
    params_features: dict,
    temp_dir: Path,
) -> pd.DataFrame:
    """Extract ROIs and compute features for a single audio file.

    Returns a DataFrame of features (one row per ROI), or empty DataFrame on failure.
    """
    # Stage 1a: Extract ROIs — saves .wav files for each ROI
    from bambird.segmentation import single_file_extract_rois
    df_rois = single_file_extract_rois(
        audio_path=str(audio_path),
        fun=params_extract["FUNC"],
        params=params_extract,
        save_path=str(temp_dir),
        display=False,
        verbose=False,
    )

    if df_rois.empty:
        return pd.DataFrame()

    # Stage 1b: Compute wavelet shape features for each ROI .wav file
    feature_frames = []
    for _, row in df_rois.iterrows():
        if "fullfilename_ts" not in row or not Path(str(row["fullfilename_ts"])).exists():
            continue
        from bambird.features import compute_features
        df_feat = compute_features(
            audio_path=str(row["fullfilename_ts"]),
            params=params_features,
            display=False,
            verbose=False,
        )
        if not df_feat.empty:
            # Carry over metadata from ROI extraction
            for col in ["filename", "fullfilename", "species", "categories", "abs_min_t"]:
                if col in row.index:
                    df_feat[col] = row[col]
            # Carry over ROI time/freq bounds
            for col in ["min_t", "max_t", "min_f", "max_f"]:
                if col in row.index:
                    df_feat[col] = row[col]
            feature_frames.append(df_feat)

    if not feature_frames:
        return pd.DataFrame()

    return pd.concat(feature_frames, ignore_index=True)


def _file_output_name(audio_path: str) -> str:
    """Derive a unique output filename from an audio path (species/stem)."""
    p = Path(audio_path)
    return f"{p.parent.name}__{p.stem}.parquet"


@app.command("extract-features")
def extract_features(
    config: str = "configs/teacher.yaml",
    partition_id: int = typer.Argument(..., help="Partition ID (0-indexed)"),
):
    """Extract ROIs and compute features for a partition of focal audio files.

    Writes one parquet per audio file for idempotency — already-processed
    files are skipped on restart.
    """
    cfg = load_config(config)
    audio_dir = cfg["data"]["focal_dir"]
    num_partitions = cfg["partitions"]["focal"]
    bambird_cfg_path = cfg["models"]["bambird"]["config"]

    output_dir = Path(cfg["output"]["root"]).expanduser() / "bambird" / "features" / f"part_{partition_id:04d}"
    ensure_output_dir(output_dir)

    # Load bambird config
    params = _load_bambird_config(bambird_cfg_path)
    params_extract = params["PARAMS_EXTRACT"]
    params_features = params["PARAMS_FEATURES"]

    files = list_audio_files(audio_dir, "focal")
    partition = partition_files(files, num_partitions, partition_id)

    # Skip already-processed files
    existing = set(p.name for p in output_dir.glob("*.parquet"))
    to_process = [(i, f) for i, f in enumerate(partition) if _file_output_name(str(f)) not in existing]

    print(f"BamBird extract-features partition {partition_id}/{num_partitions}")
    print(f"  Total files: {len(files)}, this partition: {len(partition)}")
    print(f"  Already done: {len(partition) - len(to_process)}, remaining: {len(to_process)}")

    start = time.time()
    total_rois = 0

    for idx, (i, fpath) in enumerate(to_process):
        temp_dir = Path(tempfile.mkdtemp(prefix="bambird_rois_"))
        try:
            df_feat = _extract_features_single_file(
                audio_path=str(fpath),
                params_extract=params_extract,
                params_features=params_features,
                temp_dir=temp_dir,
            )
            out_path = output_dir / _file_output_name(str(fpath))
            if not df_feat.empty:
                df_feat.to_parquet(out_path, index=False)
                total_rois += len(df_feat)
            else:
                # Write empty parquet so we don't re-process silent/empty files
                pd.DataFrame().to_parquet(out_path, index=False)
        except Exception:
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            remaining = (len(to_process) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx + 1}/{len(to_process)}] {total_rois} ROIs, {rate:.1f} files/s, ~{remaining:.0f}s left")

    elapsed = time.time() - start
    n_done = len(list(output_dir.glob("*.parquet")))
    print(f"  Done in {elapsed:.1f}s — {n_done}/{len(partition)} files complete, {total_rois} new ROIs")


@app.command("cluster")
def cluster(
    config: str = "configs/teacher.yaml",
):
    """Cluster all extracted features per-species and produce final labels."""
    cfg = load_config(config)
    bambird_cfg_path = cfg["models"]["bambird"]["config"]

    label_dir = Path(cfg["output"]["root"]).expanduser() / "bambird"
    features_dir = label_dir / "features"
    output_path = label_dir / "train_audio" / "bambird_labels.parquet"

    if output_path.exists():
        print(f"Output already exists: {output_path}, skipping.")
        return

    # Load bambird config
    params = _load_bambird_config(bambird_cfg_path)
    params_cluster = params["PARAMS_CLUSTER"]

    # Load all per-file feature parquets from partition subdirectories
    all_parquets = sorted(features_dir.glob("part_*/*.parquet"))
    if not all_parquets:
        print(f"No feature parquets found in {features_dir}/part_*/")
        return

    frames = []
    for p in all_parquets:
        df = pd.read_parquet(p)
        if not df.empty:
            frames.append(df)

    print(f"Loaded {len(all_parquets)} files ({len(frames)} non-empty)")
    if not frames:
        print("All parquets are empty — no ROIs to cluster.")
        ensure_output_dir(output_path.parent)
        pd.DataFrame(columns=["file", "species", "detected_events", "event_cluster",
                               "num_signal_rois", "num_noise_rois"]).to_parquet(output_path, index=False)
        return

    df_all = pd.concat(frames, ignore_index=True)
    print(f"  Total ROIs: {len(df_all)}")

    if df_all.empty:
        print("No ROIs to cluster.")
        ensure_output_dir(output_path.parent)
        pd.DataFrame(columns=["file", "species", "detected_events", "event_cluster",
                               "num_signal_rois", "num_noise_rois"]).to_parquet(output_path, index=False)
        return

    # Ensure required columns exist
    if "categories" not in df_all.columns:
        print("WARNING: 'categories' column missing from features — cannot cluster per-species")
        return
    if "filename_ts" not in df_all.columns:
        # compute_features adds filename_ts; if missing, create a placeholder
        df_all["filename_ts"] = df_all.index.astype(str)
    if "fullfilename_ts" not in df_all.columns:
        df_all["fullfilename_ts"] = ""

    # Run bambird clustering (per-species via 'categories' column)
    from bambird.cluster import find_cluster
    print("Clustering per-species...")
    start = time.time()
    df_cluster, _ = find_cluster(
        dataset=df_all,
        params=params_cluster,
        display=False,
        verbose=True,
    )
    elapsed = time.time() - start
    print(f"  Clustering done in {elapsed:.1f}s")

    # Convert clustered ROIs to per-file output format
    # Group by original audio file and collect events + cluster labels
    if "fullfilename" not in df_cluster.columns:
        print("WARNING: 'fullfilename' column missing — cannot group by file")
        return

    records = []
    for fullfilename, group in df_cluster.groupby("fullfilename"):
        species = group["categories"].iloc[0] if "categories" in group.columns else ""

        events = []
        clusters = []
        for _, row in group.iterrows():
            # Compute absolute time if abs_min_t is available
            abs_offset = float(row.get("abs_min_t", 0) or 0)
            min_t = float(row.get("min_t", 0)) + abs_offset
            max_t = float(row.get("max_t", 0)) + abs_offset
            events.append([round(min_t, 4), round(max_t, 4)])
            clusters.append(int(row.get("cluster_number", -1)))

        auto_labels = group["auto_label"].astype(int).tolist() if "auto_label" in group.columns else []
        num_signal = sum(1 for a in auto_labels if a == 1)
        num_noise = sum(1 for a in auto_labels if a == 0)

        records.append({
            "file": str(fullfilename),
            "species": species,
            "detected_events": json.dumps(events),
            "event_cluster": json.dumps(clusters),
            "num_signal_rois": num_signal,
            "num_noise_rois": num_noise,
        })

    df_output = pd.DataFrame(records)
    ensure_output_dir(output_path.parent)
    df_output.to_parquet(output_path, index=False)

    print(f"\nFinal output: {output_path}")
    print(f"  Files: {len(df_output)}")
    print(f"  Signal ROIs: {df_output['num_signal_rois'].sum()}")
    print(f"  Noise ROIs: {df_output['num_noise_rois'].sum()}")


def main():
    app()


if __name__ == "__main__":
    main()
