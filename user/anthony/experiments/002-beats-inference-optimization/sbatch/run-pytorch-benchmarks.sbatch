#!/bin/bash
#SBATCH --job-name=beats-bench-002
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:L40s
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/pytorch-benchmarks-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Running BEATs PyTorch benchmarks"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026
output_dir=$scratch_dir/outputs/exp002-benchmarks

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

# Run benchmark script
uv run python << 'EOF'
"""BEATs PyTorch inference benchmarks for experiment 002."""

import json
import os
from pathlib import Path

import torch

from birdclef.profiling import (
    ProfiledBEATs,
    run_benchmark,
    validate_profiling_overhead,
    reset_memory_stats,
)

# Paths
scratch_dir = Path(os.path.expanduser("~/scratch/birdclef-2026"))
checkpoint_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
output_dir = scratch_dir / "outputs/exp002-benchmarks"
output_dir.mkdir(parents=True, exist_ok=True)

# Configuration
GPU_BATCH_SIZES = [1, 4, 16, 32, 64]
CPU_BATCH_SIZES = [1, 4, 16, 32]
NUM_SAMPLES = 80000  # 5 seconds at 16kHz
WARMUP_ITERS = 10
BENCH_ITERS = 50

results = {
    "gpu": {},
    "cpu": {},
    "metadata": {
        "num_samples": NUM_SAMPLES,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
        "checkpoint": str(checkpoint_path),
    },
}

# ============================================================
# GPU Benchmarks
# ============================================================
if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("GPU BENCHMARKS")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    model_gpu = ProfiledBEATs(checkpoint_path, device="cuda")

    # Validate profiling overhead
    print("\nValidating profiling overhead...")
    profiled_ms, unprofiled_ms, overhead_pct = validate_profiling_overhead(
        model_gpu, batch_size=4, device="cuda"
    )
    print(f"  Profiled: {profiled_ms:.2f} ms")
    print(f"  Unprofiled: {unprofiled_ms:.2f} ms")
    print(f"  Overhead: {overhead_pct:.2f}%")
    results["gpu"]["overhead_validation"] = {
        "profiled_ms": profiled_ms,
        "unprofiled_ms": unprofiled_ms,
        "overhead_pct": overhead_pct,
    }

    if overhead_pct > 5:
        print(f"  WARNING: Overhead exceeds 5% target!")

    # Run benchmarks for each batch size
    for batch_size in GPU_BATCH_SIZES:
        print(f"\nBatch size: {batch_size}")
        reset_memory_stats("cuda")

        result = run_benchmark(
            model_gpu,
            batch_size=batch_size,
            num_samples=NUM_SAMPLES,
            warmup_iters=WARMUP_ITERS,
            bench_iters=BENCH_ITERS,
            device="cuda",
        )

        results["gpu"][f"batch_{batch_size}"] = result.to_dict()

        print(f"  Total: {result.timings.total_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"  Breakdown:")
        for stage, pct in result.timings.breakdown_pct().items():
            print(f"    {stage}: {pct:.1f}%")
        print(f"  Peak GPU: {result.memory.peak_gpu_mb:.1f} MB")

    del model_gpu
    torch.cuda.empty_cache()
else:
    print("No GPU available, skipping GPU benchmarks")

# ============================================================
# CPU Benchmarks
# ============================================================
print("\n" + "=" * 60)
print("CPU BENCHMARKS")
print("=" * 60)

model_cpu = ProfiledBEATs(checkpoint_path, device="cpu")

# Set thread count for fair comparison
torch.set_num_threads(4)
print(f"Using {torch.get_num_threads()} CPU threads")

# Validate profiling overhead on CPU
print("\nValidating profiling overhead...")
profiled_ms, unprofiled_ms, overhead_pct = validate_profiling_overhead(
    model_cpu, batch_size=4, device="cpu", iters=10
)
print(f"  Profiled: {profiled_ms:.2f} ms")
print(f"  Unprofiled: {unprofiled_ms:.2f} ms")
print(f"  Overhead: {overhead_pct:.2f}%")
results["cpu"]["overhead_validation"] = {
    "profiled_ms": profiled_ms,
    "unprofiled_ms": unprofiled_ms,
    "overhead_pct": overhead_pct,
}

# Run benchmarks for each batch size (fewer iters for CPU)
for batch_size in CPU_BATCH_SIZES:
    print(f"\nBatch size: {batch_size}")
    reset_memory_stats("cpu")

    result = run_benchmark(
        model_cpu,
        batch_size=batch_size,
        num_samples=NUM_SAMPLES,
        warmup_iters=3,
        bench_iters=10,  # Fewer iterations for CPU
        device="cpu",
    )

    results["cpu"][f"batch_{batch_size}"] = result.to_dict()

    print(f"  Total: {result.timings.total_ms:.2f} ms")
    print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
    print(f"  Breakdown:")
    for stage, pct in result.timings.breakdown_pct().items():
        print(f"    {stage}: {pct:.1f}%")
    print(f"  Peak RAM: {result.memory.peak_ram_mb:.1f} MB")

# ============================================================
# Save Results
# ============================================================
output_path = output_dir / "pytorch_benchmarks.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Print summary table
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Device':<8} {'Batch':<8} {'Total (ms)':<12} {'Throughput':<15} {'Transformer %':<15}")
print("-" * 58)

if "gpu" in results and results["gpu"]:
    for key, val in results["gpu"].items():
        if key.startswith("batch_"):
            batch = key.split("_")[1]
            total = val["timings"]["total_ms"]
            tput = val["throughput_samples_per_sec"]
            xfmr = val["timings_breakdown_pct"].get("transformer_pct", 0)
            print(f"{'GPU':<8} {batch:<8} {total:<12.2f} {tput:<15.1f} {xfmr:<15.1f}")

for key, val in results["cpu"].items():
    if key.startswith("batch_"):
        batch = key.split("_")[1]
        total = val["timings"]["total_ms"]
        tput = val["throughput_samples_per_sec"]
        xfmr = val["timings_breakdown_pct"].get("transformer_pct", 0)
        print(f"{'CPU':<8} {batch:<8} {total:<12.2f} {tput:<15.1f} {xfmr:<15.1f}")

print("\nDone!")
EOF

echo ""
echo "Benchmark complete!"
date
