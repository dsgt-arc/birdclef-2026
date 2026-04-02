#!/usr/bin/env python3
"""Validate Perch ONNX model against TF SavedModel.

Runs the same audio through both backends and verifies numerical equivalence.
Also benchmarks latency.

Usage:
    python scripts/validate_perch_onnx.py
    python scripts/validate_perch_onnx.py --n-samples 50
"""

import argparse
import time
from pathlib import Path

import numpy as np

# Paths
ONNX_MODEL_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-onnx")
TF_MODEL_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu")
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")

SR = 32000
WINDOW_SAMPLES = SR * 5  # 160000 samples = 5s at 32kHz


def load_tf_model(model_dir: Path):
    """Load Perch v2 TF SavedModel."""
    import tensorflow as tf

    model = tf.saved_model.load(str(model_dir))
    if hasattr(model, "infer_tf"):
        return model.infer_tf
    sigs = list(model.signatures.keys())
    if "serving_default" in sigs:
        return model.signatures["serving_default"]
    raise RuntimeError(f"Cannot find inference fn. Signatures: {sigs}")


def run_tf_inference(infer_fn, waveform_batch: np.ndarray):
    """Run TF inference. Returns (embeddings, logits)."""
    import tensorflow as tf

    input_tensor = tf.constant(waveform_batch, dtype=tf.float32)
    try:
        output = infer_fn(inputs=input_tensor)
    except TypeError:
        output = infer_fn(input_tensor)

    if isinstance(output, dict):
        emb = logits = None
        for _k, v in output.items():
            arr = v.numpy()
            if arr.shape[-1] == 1536:
                emb = arr
            elif arr.shape[-1] > 10000:
                logits = arr
        return emb.astype(np.float32), logits.astype(np.float32)
    raise RuntimeError(f"Unexpected output type: {type(output)}")


def load_onnx_model(onnx_path: Path):
    """Load Perch ONNX model."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    return session


def run_onnx_inference(session, waveform_batch: np.ndarray):
    """Run ONNX inference. Returns (embeddings, logits)."""
    outputs = session.run(None, {"inputs": waveform_batch.astype(np.float32)})
    # Output order: embedding (B, 1536), spatial_embedding, spectrogram, label (B, 14795)
    embeddings = outputs[0].astype(np.float32)
    logits = outputs[3].astype(np.float32)  # label output
    return embeddings, logits


def load_test_audio(n_samples: int) -> np.ndarray:
    """Load real audio windows for testing."""
    import soundfile as sf

    soundscape_dir = DATA_DIR / "train_soundscapes"
    audio_files = sorted(soundscape_dir.glob("*.ogg"))[:3]

    windows = []
    for audio_path in audio_files:
        audio, file_sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # Split into 5s windows
        for i in range(12):
            start = i * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            if end <= len(audio):
                windows.append(audio[start:end])
            if len(windows) >= n_samples:
                break
        if len(windows) >= n_samples:
            break

    return np.stack(windows[:n_samples], axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=12, help="Number of 5s windows to test")
    parser.add_argument("--benchmark-runs", type=int, default=5, help="Number of benchmark runs")
    args = parser.parse_args()

    print("=" * 60)
    print("Perch ONNX Validation")
    print("=" * 60)

    # Load test audio
    print(f"\nLoading {args.n_samples} test windows...")
    waveforms = load_test_audio(args.n_samples)
    print(f"  Shape: {waveforms.shape}")

    # --- TF inference ---
    print("\n--- TF SavedModel ---")
    print(f"  Loading from {TF_MODEL_DIR}")
    tf_infer = load_tf_model(TF_MODEL_DIR)

    t0 = time.time()
    tf_emb, tf_logits = run_tf_inference(tf_infer, waveforms)
    tf_time = time.time() - t0
    print(f"  Embeddings: {tf_emb.shape}, Logits: {tf_logits.shape}")
    print(f"  Time: {tf_time:.3f}s ({tf_time / len(waveforms) * 1000:.1f}ms/sample)")

    # --- ONNX inference (standard) ---
    onnx_standard = ONNX_MODEL_DIR / "perch_v2.onnx"
    print("\n--- ONNX (standard) ---")
    print(f"  Loading from {onnx_standard}")
    onnx_sess = load_onnx_model(onnx_standard)

    t0 = time.time()
    onnx_emb, onnx_logits = run_onnx_inference(onnx_sess, waveforms)
    onnx_time = time.time() - t0
    print(f"  Embeddings: {onnx_emb.shape}, Logits: {onnx_logits.shape}")
    print(f"  Time: {onnx_time:.3f}s ({onnx_time / len(waveforms) * 1000:.1f}ms/sample)")

    # --- Numerical comparison ---
    print("\n--- Numerical Comparison (TF vs ONNX standard) ---")
    emb_diff = np.abs(tf_emb - onnx_emb)
    logits_diff = np.abs(tf_logits - onnx_logits)
    print(f"  Embeddings max diff: {emb_diff.max():.10f}")
    print(f"  Embeddings mean diff: {emb_diff.mean():.10f}")
    print(f"  Logits max diff: {logits_diff.max():.10f}")
    print(f"  Logits mean diff: {logits_diff.mean():.10f}")

    emb_ok = emb_diff.max() < 1e-4
    logits_ok = logits_diff.max() < 1e-4
    print(f"  Embeddings match (tol=1e-4): {'PASS' if emb_ok else 'FAIL'}")
    print(f"  Logits match (tol=1e-4): {'PASS' if logits_ok else 'FAIL'}")

    # --- ONNX inference (no_dft) ---
    onnx_nodft = ONNX_MODEL_DIR / "perch_v2_no_dft.onnx"
    if onnx_nodft.exists():
        print("\n--- ONNX (no_dft) ---")
        print(f"  Loading from {onnx_nodft}")
        nodft_sess = load_onnx_model(onnx_nodft)

        t0 = time.time()
        nodft_emb, nodft_logits = run_onnx_inference(nodft_sess, waveforms)
        nodft_time = time.time() - t0
        print(f"  Time: {nodft_time:.3f}s ({nodft_time / len(waveforms) * 1000:.1f}ms/sample)")

        nodft_emb_diff = np.abs(tf_emb - nodft_emb)
        nodft_logits_diff = np.abs(tf_logits - nodft_logits)
        print(f"  Embeddings max diff vs TF: {nodft_emb_diff.max():.10f}")
        print(f"  Logits max diff vs TF: {nodft_logits_diff.max():.10f}")
        nodft_ok = nodft_logits_diff.max() < 5e-4
        print(f"  Match (tol=5e-4): {'PASS' if nodft_ok else 'FAIL'}")

    # --- Benchmark ---
    print(f"\n--- Benchmark ({args.benchmark_runs} runs, {len(waveforms)} samples) ---")
    times_tf = []
    times_onnx = []
    for _ in range(args.benchmark_runs):
        t0 = time.time()
        run_tf_inference(tf_infer, waveforms)
        times_tf.append(time.time() - t0)

        t0 = time.time()
        run_onnx_inference(onnx_sess, waveforms)
        times_onnx.append(time.time() - t0)

    print(f"  TF:   mean={np.mean(times_tf):.3f}s, std={np.std(times_tf):.3f}s")
    print(f"  ONNX: mean={np.mean(times_onnx):.3f}s, std={np.std(times_onnx):.3f}s")
    print(f"  Speedup: {np.mean(times_tf) / np.mean(times_onnx):.1f}x")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = emb_ok and logits_ok
    print(f"  Numerical equivalence: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Speedup: {np.mean(times_tf) / np.mean(times_onnx):.1f}x")
    print(
        f"  Recommendation: {'Safe to use ONNX' if all_pass else 'DO NOT use ONNX — outputs differ'}"
    )


if __name__ == "__main__":
    main()
