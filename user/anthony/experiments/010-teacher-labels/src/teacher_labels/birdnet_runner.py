"""BirdNET runner: species confidence + embeddings per 3s window.

Uses bioacoustics-model-zoo (bmz) for inference, which loads BirdNET's
TFLite model internally. Produces embed + predict parquet files per partition.

Output structure:
  parts/embed/part_NNNN.parquet  — 1024-dim embeddings per clip
  parts/predict/part_NNNN.parquet — species confidence scores per clip
"""

import json
import time
from pathlib import Path

import bioacoustics_model_zoo as bmz
import typer

from teacher_labels.utils import (
    list_audio_files,
    load_config,
    partition_files,
    ensure_output_dir,
)

app = typer.Typer()


@app.command()
def run(
    config: str = "configs/teacher.yaml",
    source: str = typer.Argument(..., help="'focal' or 'soundscape'"),
    partition_id: int = typer.Argument(..., help="Partition ID (0-indexed)"),
):
    """Run BirdNET inference on a partition of audio files."""
    cfg = load_config(config)
    source_key = "focal_dir" if source == "focal" else "soundscape_dir"
    audio_dir = cfg["data"][source_key]
    num_partitions = cfg["partitions"][source]
    birdnet_cfg = cfg["models"]["birdnet"]
    clip_step = birdnet_cfg["clip_step"]

    source_label = "train_audio" if source == "focal" else "train_soundscapes"
    output_dir = Path(cfg["output"]["root"]).expanduser() / "birdnet" / source_label / "parts"
    embed_path = output_dir / "embed" / f"part_{partition_id:04d}.parquet"
    predict_path = output_dir / "predict" / f"part_{partition_id:04d}.parquet"
    timing_path = output_dir / "timing" / f"part_{partition_id:04d}.json"

    if embed_path.exists() and predict_path.exists():
        print(f"Output already exists: {embed_path}, skipping.")
        return

    files = list_audio_files(audio_dir, source)
    partition = partition_files(files, num_partitions, partition_id)

    if not partition:
        print(f"No files for partition {partition_id}. Skipping.")
        return

    print(f"BirdNET {source} partition {partition_id}/{num_partitions}")
    print(f"  Total files: {len(files)}, this partition: {len(partition)}")
    print(f"  Clip step: {clip_step}s")

    # Initialize model
    model = bmz.list_models()["BirdNET"]()

    # Run inference
    start = time.time()
    embed_df, predict_df = model.embed(
        [p.as_posix() for p in partition],
        return_preds=True,
        clip_step=clip_step,
    )
    elapsed = time.time() - start

    # Save outputs
    for p in [embed_path, predict_path, timing_path]:
        ensure_output_dir(p.parent)

    embed_df.reset_index().to_parquet(embed_path, index=False)
    predict_df.reset_index().to_parquet(predict_path, index=False)

    with open(timing_path, "w") as f:
        json.dump({
            "partition_id": partition_id,
            "num_files": len(partition),
            "elapsed_s": round(elapsed, 2),
            "clip_step": clip_step,
        }, f, indent=2)

    print(f"  Done in {elapsed:.1f}s — {len(partition)} files")
    print(f"  Embeddings: {embed_df.shape}, Predictions: {predict_df.shape}")
    print(f"  Output: {output_dir}")


def main():
    app()


if __name__ == "__main__":
    main()
