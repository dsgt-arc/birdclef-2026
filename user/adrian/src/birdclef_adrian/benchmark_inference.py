"""Benchmark inference timing for OpenSoundscape and BEATs models."""

import argparse
import json
import time
from pathlib import Path

print("[DEBUG] Importing numpy...")
import numpy as np
print("[DEBUG] Importing torch...")
import torch
print("[DEBUG] Importing torchaudio...")
import torchaudio

# OpenSoundscape models - optional for now
print("[DEBUG] Trying opensoundscape import...")
try:
    from opensoundscape.torch.models.cnn import load_model as load_oss_model
    OPENSOUNDSCAPE_AVAILABLE = True
    print("[DEBUG] OpenSoundscape available")
except ImportError:
    OPENSOUNDSCAPE_AVAILABLE = False
    print("[DEBUG] OpenSoundscape not available")

# BEATs encoder from birdclef-anthony
print("[DEBUG] Importing BEATsEncoder...")
from birdclef.encoders import BEATsEncoder
print("[DEBUG] All imports complete")



def generate_noise_audio(
    duration_seconds: float = 60.0,
    sample_rate: int = 16000,
    output_path: Path | None = None,
) -> torch.Tensor:
    """Generate white noise audio for benchmarking.

    Args:
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate in Hz.
        output_path: Optional path to save the audio file.

    Returns:
        Audio tensor of shape (1, num_samples).
    """
    num_samples = int(duration_seconds * sample_rate)
    noise = torch.randn(1, num_samples) * 0.1  # Scale down to avoid clipping

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), noise, sample_rate)
        print(f"Saved noise audio to {output_path}")

    return noise


def benchmark_opensoundscape(
    model_name: str,
    audio: torch.Tensor,
    sample_rate: int,
    num_runs: int = 10,
    warmup_runs: int = 2,
    device: str = "cuda",
) -> dict:
    """Benchmark OpenSoundscape model inference.

    Args:
        model_name: Name of the prebuilt model (e.g., 'BirdNET').
        audio: Audio tensor of shape (1, num_samples).
        sample_rate: Sample rate in Hz.
        num_runs: Number of benchmark runs.
        warmup_runs: Number of warmup runs.
        device: Device to run inference on.

    Returns:
        Dictionary with timing results.
    """
    if not OPENSOUNDSCAPE_AVAILABLE:
        raise ImportError(f"OpenSoundscape not available. Cannot benchmark {model_name}")

    print(f"\nBenchmarking OpenSoundscape model: {model_name}")

    # Load model
    load_start = time.perf_counter()
    model = load_oss_model(model_name)
    load_time = time.perf_counter() - load_start
    print(f"  Model loaded in {load_time:.3f}s")

    # Save audio temporarily for opensoundscape
    tmp_audio_path = Path("/tmp/benchmark_audio.wav")
    torchaudio.save(str(tmp_audio_path), audio, sample_rate)

    # Warmup runs
    print(f"  Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = model.predict([tmp_audio_path])

    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = model.predict([tmp_audio_path])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.4f}s")

    # Clean up
    tmp_audio_path.unlink(missing_ok=True)

    return {
        "model": model_name,
        "type": "opensoundscape",
        "load_time_s": load_time,
        "mean_inference_s": float(np.mean(times)),
        "std_inference_s": float(np.std(times)),
        "min_inference_s": float(np.min(times)),
        "max_inference_s": float(np.max(times)),
        "num_runs": num_runs,
    }


def benchmark_beats(
    checkpoint_path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    num_runs: int = 10,
    warmup_runs: int = 2,
    device: str = "cuda",
) -> dict:
    """Benchmark BEATs model inference.

    Args:
        checkpoint_path: Path to BEATs checkpoint.
        audio: Audio tensor of shape (1, num_samples).
        sample_rate: Sample rate in Hz.
        num_runs: Number of benchmark runs.
        warmup_runs: Number of warmup runs.
        device: Device to run inference on.

    Returns:
        Dictionary with timing results.
    """
    print(f"\nBenchmarking BEATs model")

    # Load model
    load_start = time.perf_counter()
    encoder = BEATsEncoder(checkpoint_path=checkpoint_path, device=device)
    load_time = time.perf_counter() - load_start
    print(f"  Model loaded in {load_time:.3f}s")

    # Prepare audio (add batch dimension)
    audio_batch = audio.unsqueeze(0).to(device)  # (1, 1, num_samples)

    # Warmup runs
    print(f"  Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = encoder.extract_features(audio_batch)

    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            _ = encoder.extract_features(audio_batch)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.4f}s")

    return {
        "model": "BEATs",
        "type": "beats",
        "embed_dim": encoder.embed_dim,
        "load_time_s": load_time,
        "mean_inference_s": float(np.mean(times)),
        "std_inference_s": float(np.std(times)),
        "min_inference_s": float(np.min(times)),
        "max_inference_s": float(np.max(times)),
        "num_runs": num_runs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference timing for audio models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["BirdNET", "Perch", "beats"],
        help="Models to benchmark (e.g., BirdNET, Perch, beats)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Audio duration in seconds",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.home() / "scratch" / "birdclef-2026" / "checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "scratch" / "birdclef-2026" / "outputs" / "001-inference-benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with minimal iterations for testing",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.num_runs = 2
        args.warmup_runs = 1
        args.duration = 5.0
        print("Dry run mode: minimal iterations")

    print("=" * 60)
    print("Inference Benchmark Suite")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")

    # Generate noise audio
    audio = generate_noise_audio(
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
    )

    results = []

    for model_name in args.models:
        try:
            if model_name.lower() == "beats":
                ckpt_path = args.checkpoint_dir / "beats" / "BEATs_iter3_plus_AS2M.pt"
                if not ckpt_path.exists():
                    print(f"  WARNING: BEATs checkpoint not found at {ckpt_path}")
                    continue
                result = benchmark_beats(
                    checkpoint_path=ckpt_path,
                    audio=audio,
                    sample_rate=args.sample_rate,
                    num_runs=args.num_runs,
                    warmup_runs=args.warmup_runs,
                    device=args.device,
                )
            else:
                result = benchmark_opensoundscape(
                    model_name=model_name,
                    audio=audio,
                    sample_rate=args.sample_rate,
                    num_runs=args.num_runs,
                    warmup_runs=args.warmup_runs,
                    device=args.device,
                )
            results.append(result)
        except Exception as e:
            print(f"  ERROR benchmarking {model_name}: {e}")
            results.append({"model": model_name, "error": str(e)})

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "duration_s": args.duration,
                    "sample_rate": args.sample_rate,
                    "num_runs": args.num_runs,
                    "warmup_runs": args.warmup_runs,
                    "device": args.device,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Mean (s)':<12} {'Std (s)':<12} {'Load (s)':<12}")
    print("-" * 51)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<15} ERROR")
        else:
            print(
                f"{r['model']:<15} {r['mean_inference_s']:<12.4f} "
                f"{r['std_inference_s']:<12.4f} {r['load_time_s']:<12.3f}"
            )


if __name__ == "__main__":
    main()
