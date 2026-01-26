"""Benchmark inference timing for OpenSoundscape and BEATs models."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

# Bioacoustics Model Zoo
try:
    import bioacoustics_model_zoo as bmz
    BMZ_AVAILABLE = True
except ImportError:
    BMZ_AVAILABLE = False

from birdclef.encoders import BEATsEncoder



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
        # Use soundfile directly to avoid torchaudio/torchcodec issues
        sf.write(str(output_path), noise.squeeze().numpy(), sample_rate)
        # print(f"Saved noise audio to {output_path}")

    return noise


def benchmark_bmz_model(
    model_name: str,
    audio_path: Path,
    num_runs: int = 10,
    warmup_runs: int = 2,
    device: str = "cuda",
) -> dict:
    """Benchmark Bioacoustics Model Zoo model inference.

    Args:
        model_name: Name of the BMZ model class.
        audio_path: Path to audio file.
        num_runs: Number of benchmark runs.
        warmup_runs: Number of warmup runs.
        device: Device to run inference on.

    Returns:
        Dictionary with timing results.
    """
    print(f"\nBenchmarking BMZ model: {model_name}")

    if not BMZ_AVAILABLE:
        raise ImportError("bioacoustics_model_zoo is not installed")

    # Load model
    load_start = time.perf_counter()
    model_cls = getattr(bmz, model_name)
    # Instantiate model
    try:
        model = model_cls()
    except Exception as e:
        # Some models might need arguments or fail to load if deps are missing
        raise RuntimeError(f"Failed to instantiate {model_name}: {e}")

    # Move to device if supported
    if hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            pass # Ignore if model doesn't support moving to device (e.g. TF/TFLite wrappers)

    load_time = time.perf_counter() - load_start
    print(f"  Model loaded in {load_time:.3f}s")
    
    # Warmup runs
    print(f"  Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = model.embed([audio_path])

    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = model.embed([audio_path])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.4f}s")

    return {
        "model": model_name,
        "type": "bmz",
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
        default=["bmz", "beats"],
        help="Models to benchmark. Use 'bmz', 'all' for all BMZ models, or specific model names.",
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
    
    # Expand model list
    model_list = []
    available_bmz_models = bmz.list_models() if BMZ_AVAILABLE else []
    
    for m in args.models:
        if m.lower() == 'all' or m.lower() == 'bmz':
            if BMZ_AVAILABLE:
                model_list.extend(available_bmz_models)
            else:
                print("WARNING: bioacoustics_model_zoo not available.")
        elif m.lower() == 'beats':
            model_list.append('BEATs')
        elif m in available_bmz_models:
            model_list.append(m)
        else:
             # Fallback if user passes specific name but we missed it, or just add it
             model_list.append(m)
             
    # Deduplicate
    model_list = sorted(list(set(model_list)))
    print(f"Models to benchmark: {model_list}")

    # Generate noise audio
    tmp_audio_path = Path("/tmp") / f"benchmark_noise_{os.getpid()}.wav"
    audio = generate_noise_audio(
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        output_path=tmp_audio_path
    )

    results = []

    for model_name in model_list:
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
                # Assume BMZ model
                result = benchmark_bmz_model(
                    model_name=model_name,
                    audio_path=tmp_audio_path,
                    num_runs=args.num_runs,
                    warmup_runs=args.warmup_runs,
                    device=args.device,
                )
            results.append(result)
        except Exception as e:
            print(f"  ERROR benchmarking {model_name}: {e}")
            results.append({"model": model_name, "error": str(e)})

    # Cleanup
    if tmp_audio_path.exists():
        tmp_audio_path.unlink()


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
