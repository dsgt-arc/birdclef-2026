#!/bin/bash
#SBATCH --job-name=static-inf-002
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t60 -qembers
#SBATCH --output=logs/static-inference-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Running static embeddings inference benchmarks"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python << 'EOF'
"""Benchmark static embeddings inference pipeline."""

import json
import os
import sys
import time
from pathlib import Path

print("Loading numpy...", flush=True)
import numpy as np

print("Loading onnxruntime...", flush=True)
import onnxruntime as ort

print("Loading torch...", flush=True)
import torch

sys.path.insert(0, "vendor/unilm/beats")

print("Loading static embeddings module...", flush=True)
from birdclef.static_embeddings import StaticEmbeddingsPipeline, benchmark_static_pipeline, load_beats_encoder

# Paths
scratch_dir = Path(os.path.expanduser("~/scratch/birdclef-2026"))
onnx_frontend_path = scratch_dir / "outputs/exp002-benchmarks/onnx/beats_frontend.onnx"
static_dir = scratch_dir / "outputs/exp002-benchmarks/static_embeddings"
codebook_path = static_dir / "codebook.npy"
embeddings_path = static_dir / "static_embeddings.npy"
embeddings_pca_path = static_dir / "static_embeddings_pca.npy"
weights_path = static_dir / "sif_weights.npy"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
output_path = scratch_dir / "outputs/exp002-benchmarks/static_inference_benchmarks.json"
esc50_dir = scratch_dir / "data/esc50"

print("=" * 60)
print("STATIC EMBEDDINGS INFERENCE BENCHMARKS")
print("=" * 60)

# Initialize pipelines
print("\nInitializing pipelines...")

# Full dim (768) without weights
pipeline_768 = StaticEmbeddingsPipeline(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_path,
    weights_path=None,
    num_threads=4,
)
print(f"  Pipeline 768d (no weights): embed_dim={pipeline_768.embed_dim}")

# Full dim (768) with SIF weights
pipeline_768_sif = StaticEmbeddingsPipeline(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_path,
    weights_path=weights_path,
    num_threads=4,
)
print(f"  Pipeline 768d (SIF weights): embed_dim={pipeline_768_sif.embed_dim}")

# PCA dim (256) with SIF weights
pipeline_256_sif = StaticEmbeddingsPipeline(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_pca_path,
    weights_path=weights_path,
    num_threads=4,
)
print(f"  Pipeline 256d (SIF weights): embed_dim={pipeline_256_sif.embed_dim}")

results = {}

# Benchmark pipelines
print("\n" + "=" * 60)
print("BENCHMARK: Static Pipeline 768d (no weights)")
print("=" * 60)
results["static_768"] = benchmark_static_pipeline(pipeline_768)

print("\n" + "=" * 60)
print("BENCHMARK: Static Pipeline 768d (SIF weights)")
print("=" * 60)
results["static_768_sif"] = benchmark_static_pipeline(pipeline_768_sif)

print("\n" + "=" * 60)
print("BENCHMARK: Static Pipeline 256d (SIF weights)")
print("=" * 60)
results["static_256_sif"] = benchmark_static_pipeline(pipeline_256_sif)

# Benchmark full BEATs for comparison
print("\n" + "=" * 60)
print("BENCHMARK: Full BEATs (PyTorch CPU)")
print("=" * 60)

encoder = load_beats_encoder(encoder_path)
encoder.eval()

def benchmark_full_beats(encoder, batch_sizes=[1, 4, 16, 32], num_samples=80000):
    """Benchmark full BEATs encoder with waveform input."""
    beats_results = {}
    for batch_size in batch_sizes:
        # Create random waveform input (5 seconds at 16kHz)
        waveform = torch.randn(batch_size, num_samples)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = encoder.extract_features(waveform)[0].mean(dim=1)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = encoder.extract_features(waveform)[0].mean(dim=1)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        throughput = (batch_size * 1000) / avg_ms

        beats_results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "avg_ms": avg_ms,
            "throughput_samples_per_sec": throughput,
            "min_ms": min(times),
            "max_ms": max(times),
        }
        print(f"Batch {batch_size}: {avg_ms:.2f} ms, {throughput:.1f} samples/sec")

    return beats_results

results["full_beats"] = benchmark_full_beats(encoder)

# Compute speedups
print("\n" + "=" * 60)
print("SPEEDUP SUMMARY")
print("=" * 60)

print("\nSpeedup vs Full BEATs (higher is better):")
print("-" * 50)
for batch_key in ["batch_1", "batch_4", "batch_16", "batch_32"]:
    beats_ms = results["full_beats"][batch_key]["avg_ms"]
    static_768_ms = results["static_768"][batch_key]["avg_ms"]
    static_768_sif_ms = results["static_768_sif"][batch_key]["avg_ms"]
    static_256_sif_ms = results["static_256_sif"][batch_key]["avg_ms"]

    print(f"{batch_key}:")
    print(f"  Static 768d:         {beats_ms/static_768_ms:.1f}x faster ({static_768_ms:.2f} vs {beats_ms:.2f} ms)")
    print(f"  Static 768d + SIF:   {beats_ms/static_768_sif_ms:.1f}x faster ({static_768_sif_ms:.2f} vs {beats_ms:.2f} ms)")
    print(f"  Static 256d + SIF:   {beats_ms/static_256_sif_ms:.1f}x faster ({static_256_sif_ms:.2f} vs {beats_ms:.2f} ms)")

# Save results
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

print("\nBenchmarks complete!")
EOF

echo ""
echo "Static inference benchmarks complete!"
date
