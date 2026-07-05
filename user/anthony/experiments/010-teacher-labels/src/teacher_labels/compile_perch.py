"""Compile Perch model to TFLite format.

One-time step that requires full TensorFlow installation.
After compilation, runtime inference uses ai-edge-litert only.

Reference: dsgt-arc/birdclef-2025 `birdclef/kaggle/compile.py`
"""

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def compile(
    output_path: str = typer.Option(
        "~/scratch/birdclef-2026/outputs/exp010/perch.tflite",
        help="Path to save the compiled TFLite model",
    ),
):
    """Compile Perch TF model to TFLite format."""
    import bioacoustics_model_zoo as bmz
    import tensorflow as tf

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"TFLite model already exists: {output}")
        return

    print("Loading Perch model via bmz...")
    perch = bmz.list_models()["Perch"]()

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(perch.tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Flex ops required: FlexEnsureShape, FlexStridedSlice, FlexTranspose
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    with output.open("wb") as f:
        f.write(tflite_model)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Saved TFLite model: {output} ({size_mb:.1f} MB)")


def main():
    app()


if __name__ == "__main__":
    main()
