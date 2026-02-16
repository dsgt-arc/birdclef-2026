"""Perch runner: species confidence + 1280-dim embeddings per 5s window.

Uses ai-edge-litert TFLite interpreter for inference (no full TF needed).
The TFLite model MUST be compiled first via teacher-compile-perch.

Output structure:
  parts/embed/part_NNNN.parquet  — 1280-dim embeddings per clip
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
    """Run Perch inference on a partition of audio files."""
    cfg = load_config(config)
    source_key = "focal_dir" if source == "focal" else "soundscape_dir"
    audio_dir = cfg["data"][source_key]
    num_partitions = cfg["partitions"][source]
    perch_cfg = cfg["models"]["perch"]
    clip_step = perch_cfg["clip_step"]
    tflite_path = Path(perch_cfg["tflite_path"]).expanduser()

    source_label = "train_audio" if source == "focal" else "train_soundscapes"
    output_dir = Path(cfg["output"]["root"]).expanduser() / "perch" / source_label / "parts"
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

    print(f"Perch {source} partition {partition_id}/{num_partitions}")
    print(f"  Total files: {len(files)}, this partition: {len(partition)}")
    print(f"  Clip step: {clip_step}s")

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"Perch TFLite model not found at {tflite_path}\n"
            f"Run 'teacher-compile-perch' first (or sbatch sbatch/compile-perch.sbatch)"
        )

    file_paths = [p.as_posix() for p in partition]
    print(f"  Using TFLite model: {tflite_path}")
    elapsed, embed_df, predict_df = _run_tflite(tflite_path, file_paths, clip_step)

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
            "backend": "tflite",
        }, f, indent=2)

    print(f"  Done in {elapsed:.1f}s — {len(partition)} files")
    print(f"  Embeddings: {embed_df.shape}, Predictions: {predict_df.shape}")
    print(f"  Output: {output_dir}")


def _run_tflite(tflite_path: Path, file_paths: list[str], clip_step: float):
    """Run Perch via TFLite interpreter.

    Uses bmz to create the dataloader (handles audio chunking), then
    runs the TFLite model for the forward pass.
    """
    import numpy as np
    import pandas as pd
    from ai_edge_litert.interpreter import Interpreter

    # Load bmz Perch for dataloader creation only
    perch = bmz.list_models()["Perch"]()

    # Load TFLite interpreter
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start = time.time()

    # Use bmz dataloader for audio preprocessing
    dataloader = perch.predict_dataloader(file_paths, batch_size=1)

    embeddings = []
    for batch in dataloader:
        interpreter.set_tensor(input_details[0]["index"], batch[0])
        interpreter.invoke()
        # Perch: output index 1 is embeddings (1280-dim)
        embed_data = interpreter.get_tensor(output_details[1]["index"])
        embeddings.append(embed_data)

    embed_array = np.concatenate(embeddings, axis=0).squeeze()
    embed_df = pd.DataFrame(
        data=embed_array,
        index=dataloader.dataset.dataset.label_df.index,
    )

    # For predictions, run embed() with bmz to get species scores
    # (TFLite only gives raw embeddings, not species logits)
    _, predict_df = perch.embed(
        file_paths,
        return_preds=True,
        clip_step=clip_step,
    )

    elapsed = time.time() - start
    return elapsed, embed_df, predict_df


def main():
    app()


if __name__ == "__main__":
    main()
