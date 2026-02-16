"""Benchmark runner: timing + format verification for all teacher models.

Runs each model on a small sample of files, reports:
- Inference time per file and per 10s chunk
- Extrapolated wall time for full dataset at N partitions
- Output format verification (columns, dtypes)
- Estimated output size for full dataset
"""

import json
import time
from pathlib import Path

import pandas as pd
import typer

from teacher_labels.utils import list_audio_files, load_config, ensure_output_dir

app = typer.Typer()

# Approximate counts for BirdCLEF 2025
FOCAL_COUNT = 28564
SOUNDSCAPE_COUNT = 9726
TOTAL_COUNT = FOCAL_COUNT + SOUNDSCAPE_COUNT


def benchmark_bambird(files: list[Path], cfg: dict) -> dict:
    """Benchmark BamBird on sample files."""
    from teacher_labels.bambird_runner import extract_rois

    bambird_cfg = cfg["models"]["bambird"]
    results = []
    start = time.time()
    for fpath in files:
        t0 = time.time()
        events = extract_rois(
            fpath,
            min_duration=bambird_cfg["min_roi_duration"],
            max_duration=bambird_cfg["max_roi_duration"],
        )
        elapsed = time.time() - t0
        results.append({
            "file": fpath.name,
            "elapsed_s": round(elapsed, 3),
            "num_rois": len(events),
        })
    total_elapsed = time.time() - start

    return {
        "model": "bambird",
        "num_files": len(files),
        "total_elapsed_s": round(total_elapsed, 2),
        "per_file_s": round(total_elapsed / len(files), 3),
        "per_file_details": results,
        "schema": ["file", "events_json", "num_rois", "duration_s"],
    }


def benchmark_birdnet(files: list[Path], cfg: dict) -> dict:
    """Benchmark BirdNET on sample files."""
    import bioacoustics_model_zoo as bmz

    clip_step = cfg["models"]["birdnet"]["clip_step"]
    model = bmz.list_models()["BirdNET"]()

    start = time.time()
    embed_df, predict_df = model.embed(
        [p.as_posix() for p in files],
        return_preds=True,
        clip_step=clip_step,
    )
    total_elapsed = time.time() - start

    embed_size = embed_df.memory_usage(deep=True).sum()
    predict_size = predict_df.memory_usage(deep=True).sum()

    return {
        "model": "birdnet",
        "num_files": len(files),
        "total_elapsed_s": round(total_elapsed, 2),
        "per_file_s": round(total_elapsed / len(files), 3),
        "clip_step": clip_step,
        "embed_shape": list(embed_df.shape),
        "predict_shape": list(predict_df.shape),
        "embed_columns_sample": list(embed_df.columns[:5]),
        "predict_columns_sample": list(predict_df.columns[:5]),
        "embed_memory_mb": round(embed_size / 1e6, 2),
        "predict_memory_mb": round(predict_size / 1e6, 2),
    }


def benchmark_perch(files: list[Path], cfg: dict) -> dict:
    """Benchmark Perch on sample files."""
    import bioacoustics_model_zoo as bmz

    clip_step = cfg["models"]["perch"]["clip_step"]
    model = bmz.list_models()["Perch"]()

    start = time.time()
    embed_df, predict_df = model.embed(
        [p.as_posix() for p in files],
        return_preds=True,
        clip_step=clip_step,
    )
    total_elapsed = time.time() - start

    embed_size = embed_df.memory_usage(deep=True).sum()
    predict_size = predict_df.memory_usage(deep=True).sum()

    return {
        "model": "perch",
        "num_files": len(files),
        "total_elapsed_s": round(total_elapsed, 2),
        "per_file_s": round(total_elapsed / len(files), 3),
        "clip_step": clip_step,
        "embed_shape": list(embed_df.shape),
        "predict_shape": list(predict_df.shape),
        "embed_columns_sample": list(embed_df.columns[:5]),
        "predict_columns_sample": list(predict_df.columns[:5]),
        "embed_memory_mb": round(embed_size / 1e6, 2),
        "predict_memory_mb": round(predict_size / 1e6, 2),
    }


def extrapolate(result: dict, num_partitions: int) -> dict:
    """Extrapolate timing to full dataset."""
    per_file = result["per_file_s"]
    files_per_partition = TOTAL_COUNT / num_partitions
    wall_time_s = per_file * files_per_partition
    return {
        "model": result["model"],
        "per_file_s": per_file,
        "files_per_partition": round(files_per_partition),
        "wall_time_per_partition_m": round(wall_time_s / 60, 1),
        "total_wall_time_h": round(wall_time_s / 3600, 2),
    }


@app.command()
def run(
    config: str = "configs/teacher.yaml",
    num_files: int = typer.Option(None, help="Override number of files to benchmark"),
    output: str = typer.Option(None, help="Path to save benchmark JSON"),
):
    """Run benchmarks for all teacher models."""
    cfg = load_config(config)
    n = num_files or cfg["benchmark"]["num_files"]

    # Get sample files from focal recordings
    files = list_audio_files(cfg["data"]["focal_dir"], "focal")[:n]
    print(f"Benchmarking with {len(files)} files\n")

    results = {}

    # BamBird
    print("=== BamBird ===")
    try:
        results["bambird"] = benchmark_bambird(files, cfg)
        print(f"  {results['bambird']['per_file_s']}s per file")
        ext = extrapolate(results["bambird"], cfg["partitions"]["focal"])
        results["bambird_extrapolated"] = ext
        print(f"  Extrapolated: {ext['wall_time_per_partition_m']}m per partition")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["bambird"] = {"error": str(e)}

    # BirdNET
    print("\n=== BirdNET ===")
    try:
        results["birdnet"] = benchmark_birdnet(files, cfg)
        print(f"  {results['birdnet']['per_file_s']}s per file")
        print(f"  Embed: {results['birdnet']['embed_shape']}, Predict: {results['birdnet']['predict_shape']}")
        ext = extrapolate(results["birdnet"], cfg["partitions"]["focal"])
        results["birdnet_extrapolated"] = ext
        print(f"  Extrapolated: {ext['wall_time_per_partition_m']}m per partition")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["birdnet"] = {"error": str(e)}

    # Perch
    print("\n=== Perch ===")
    try:
        results["perch"] = benchmark_perch(files, cfg)
        print(f"  {results['perch']['per_file_s']}s per file")
        print(f"  Embed: {results['perch']['embed_shape']}, Predict: {results['perch']['predict_shape']}")
        ext = extrapolate(results["perch"], cfg["partitions"]["focal"])
        results["perch_extrapolated"] = ext
        print(f"  Extrapolated: {ext['wall_time_per_partition_m']}m per partition")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["perch"] = {"error": str(e)}

    # Save results
    if output:
        out_path = Path(output).expanduser()
        ensure_output_dir(out_path.parent)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print(f"\n{json.dumps(results, indent=2)}")


def main():
    app()


if __name__ == "__main__":
    main()
