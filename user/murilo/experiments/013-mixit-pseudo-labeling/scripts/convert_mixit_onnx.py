#!/usr/bin/env python3
"""Convert Bird MixIT TF1 checkpoint to ONNX format.

The pretrained Bird MixIT model (ICASSP 2022) is distributed as a TF1 MetaGraph
+ checkpoint. This script converts it to ONNX for portable inference via onnxruntime.

Input tensor:  input_audio/receiver_audio:0  — (num_samples,) mono waveform at 22.05 kHz
Output tensor: denoised_waveforms:0          — (num_sources, num_samples) separated channels

Usage:
    python scripts/convert_mixit_onnx.py \
        --checkpoint-dir /storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit \
        --output /storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx

    # Validate against TF1 output
    python scripts/convert_mixit_onnx.py \
        --checkpoint-dir /path/to/bird-mixit \
        --output /path/to/bird_mixit_4sources.onnx \
        --validate --test-audio /path/to/test.ogg

Requirements:
    uv pip install tensorflow tf2onnx onnx onnxruntime
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_tf1_to_onnx(checkpoint_dir: Path, output_path: Path):
    """Convert TF1 MetaGraph + checkpoint to ONNX."""
    import tensorflow as tf
    import tf2onnx

    meta_path = checkpoint_dir / "inference.meta"
    checkpoint_prefix = str(checkpoint_dir / "model.ckpt-3223090")

    if not meta_path.exists():
        raise FileNotFoundError(f"MetaGraph not found: {meta_path}")

    logger.info(f"Loading TF1 MetaGraph from {meta_path}")
    logger.info(f"Checkpoint prefix: {checkpoint_prefix}")

    # Load the TF1 graph
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # Import the metagraph and restore weights
        saver = tf.compat.v1.train.import_meta_graph(str(meta_path))
        saver.restore(sess, checkpoint_prefix)

        # Get the graph definition
        graph_def = sess.graph.as_graph_def()

        # Freeze the graph (replace variables with constants)
        # Use compat.v1 API — graph_util moved in TF2
        output_node_name = "denoised_waveforms"
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, graph_def, [output_node_name]
        )

    logger.info(f"Frozen graph has {len(frozen_graph.node)} nodes")

    # Convert frozen graph to ONNX
    logger.info("Converting to ONNX...")
    input_name = "input_audio/receiver_audio:0"
    output_name = "denoised_waveforms:0"

    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen_graph,
        input_names=[input_name],
        output_names=[output_name],
        opset=17,
    )

    # Save ONNX model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import onnx

    onnx.save(model_proto, str(output_path))
    logger.info(f"ONNX model saved to {output_path}")
    logger.info(f"Model size: {output_path.stat().st_size / 1e6:.1f} MB")

    return model_proto


def validate_onnx(
    checkpoint_dir: Path,
    onnx_path: Path,
    test_audio_path: Path | None = None,
    atol: float = 1e-4,
):
    """Validate ONNX output matches TF1 output."""
    import onnxruntime as ort
    import tensorflow as tf

    # Generate or load test input
    if test_audio_path is not None:
        import soundfile as sf

        audio, sr = sf.read(str(test_audio_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # Resample to 22050 if needed
        if sr != 22050:
            import resampy

            audio = resampy.resample(audio, sr, 22050)
        # Use first 5 seconds
        audio = audio[: 22050 * 5]
        logger.info(f"Test audio: {test_audio_path} ({len(audio)} samples at 22050 Hz)")
    else:
        # Use random noise as test input
        np.random.seed(42)
        audio = np.random.randn(22050 * 5).astype(np.float32) * 0.1
        logger.info("Test audio: random noise (5s at 22050 Hz)")

    # The TF1 model expects input shape (1, 1, num_samples) — batch=1, mics=1, samples=N
    # Output shape is (1, num_sources, num_samples)
    audio_3d = audio.reshape(1, 1, -1)
    logger.info(f"Input shape for inference: {audio_3d.shape}")

    # Run TF1 inference
    logger.info("Running TF1 inference...")
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(str(checkpoint_dir / "inference.meta"))
        saver.restore(sess, str(checkpoint_dir / "model.ckpt-3223090"))

        tf_output = sess.run(
            "denoised_waveforms:0",
            feed_dict={"input_audio/receiver_audio:0": audio_3d},
        )
    logger.info(f"TF1 output shape: {tf_output.shape}")

    # Run ONNX inference
    logger.info("Running ONNX inference...")
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: audio_3d})[0]
    logger.info(f"ONNX output shape: {onnx_output.shape}")

    # Squeeze batch dimension for comparison: (1, sources, samples) → (sources, samples)
    tf_out = tf_output.squeeze(0) if tf_output.ndim == 3 else tf_output
    onnx_out = onnx_output.squeeze(0) if onnx_output.ndim == 3 else onnx_output

    # Compare
    max_diff = np.max(np.abs(tf_out - onnx_out))
    mean_diff = np.mean(np.abs(tf_out - onnx_out))
    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")

    if max_diff < atol:
        logger.info(f"VALIDATION PASSED (max diff {max_diff:.6f} < atol {atol})")
    else:
        logger.warning(
            f"VALIDATION FAILED (max diff {max_diff:.6f} >= atol {atol}). "
            f"Outputs may still be usable — check if differences are negligible."
        )

    # Log channel activity (which channels have signal)
    for i in range(onnx_out.shape[0]):
        rms = np.sqrt(np.mean(onnx_out[i] ** 2))
        logger.info(f"  Channel {i}: RMS = {rms:.6f}")

    return max_diff < atol


def main():
    parser = argparse.ArgumentParser(description="Convert Bird MixIT TF1 to ONNX")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit",
        help="Directory containing inference.meta and model.ckpt-* files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX output matches TF1 output",
    )
    parser.add_argument(
        "--test-audio",
        type=str,
        default=None,
        help="Path to test audio file for validation (uses random noise if not provided)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for validation (default: 1e-4)",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)

    # Convert
    convert_tf1_to_onnx(checkpoint_dir, output_path)

    # Validate
    if args.validate:
        test_audio = Path(args.test_audio) if args.test_audio else None
        validate_onnx(checkpoint_dir, output_path, test_audio, args.atol)


if __name__ == "__main__":
    main()
