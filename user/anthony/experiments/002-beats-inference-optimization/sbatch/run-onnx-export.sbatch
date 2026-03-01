#!/bin/bash
#SBATCH --job-name=onnx-export-002
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t60 -qembers
#SBATCH --output=logs/onnx-export-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Running BEATs ONNX export and benchmarks"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python << 'EOF'
"""ONNX export and benchmarking for BEATs."""

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.onnx_export import (
    export_frontend,
    export_full_encoder,
    verify_onnx_output,
    benchmark_onnx,
    BEATsFrontend,
    BEATsEncoder,
)
from BEATs import BEATs, BEATsConfig

# Paths
scratch_dir = Path(os.path.expanduser("~/scratch/birdclef-2026"))
checkpoint_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
output_dir = scratch_dir / "outputs/exp002-benchmarks/onnx"
output_dir.mkdir(parents=True, exist_ok=True)

results = {"frontend": {}, "full_encoder": {}}

# Load PyTorch model
print("Loading PyTorch model...")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
cfg = BEATsConfig(checkpoint["cfg"])
beats = BEATs(cfg)
beats.load_state_dict(checkpoint["model"])
beats.eval()

# ============================================================
# Frontend Export
# ============================================================
print("\n" + "=" * 60)
print("FRONTEND EXPORT")
print("=" * 60)

frontend_path = output_dir / "beats_frontend.onnx"
print(f"Exporting to {frontend_path}...")
export_frontend(checkpoint_path, frontend_path)

# Verify
frontend = BEATsFrontend(beats)
frontend.eval()
test_input = torch.randn(4, 1, 499, 128)
matches, max_diff = verify_onnx_output(frontend, frontend_path, test_input)
print(f"Output matches PyTorch: {matches}, max_diff: {max_diff:.6f}")
results["frontend"]["verification"] = {"matches": matches, "max_diff": float(max_diff)}

# Benchmark
print("\nBenchmarking frontend ONNX...")
frontend_bench = benchmark_onnx(
    frontend_path,
    batch_sizes=[1, 4, 16, 32],
    warmup_iters=5,
    bench_iters=20,
    num_threads=4,
)
results["frontend"]["benchmarks"] = frontend_bench

# ============================================================
# Full Encoder Export
# ============================================================
print("\n" + "=" * 60)
print("FULL ENCODER EXPORT")
print("=" * 60)

encoder_path = output_dir / "beats_encoder.onnx"
print(f"Exporting to {encoder_path}...")
export_full_encoder(checkpoint_path, encoder_path)

# Verify
encoder = BEATsEncoder(beats)
encoder.eval()
matches, max_diff = verify_onnx_output(encoder, encoder_path, test_input)
print(f"Output matches PyTorch: {matches}, max_diff: {max_diff:.6f}")
results["full_encoder"]["verification"] = {"matches": matches, "max_diff": float(max_diff)}

# Benchmark
print("\nBenchmarking full encoder ONNX...")
encoder_bench = benchmark_onnx(
    encoder_path,
    batch_sizes=[1, 4, 16, 32],
    warmup_iters=3,
    bench_iters=10,  # Fewer iters since it's slower
    num_threads=4,
)
results["full_encoder"]["benchmarks"] = encoder_bench

# ============================================================
# Save Results
# ============================================================
results_path = output_dir / "onnx_benchmarks.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: ONNX vs PyTorch CPU")
print("=" * 60)

# Load PyTorch CPU benchmarks for comparison
pytorch_results_path = scratch_dir / "outputs/exp002-benchmarks/pytorch_benchmarks.json"
if pytorch_results_path.exists():
    with open(pytorch_results_path) as f:
        pytorch_results = json.load(f)

    print(f"\n{'Model':<20} {'Batch':<8} {'ONNX (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 64)

    for batch_key in ["batch_1", "batch_4", "batch_16", "batch_32"]:
        batch = batch_key.split("_")[1]

        # Frontend comparison (frontend vs fbank+patch_embed+layer_norm+proj in PyTorch)
        if batch_key in frontend_bench:
            onnx_ms = frontend_bench[batch_key]["avg_ms"]
            # Sum frontend stages from PyTorch
            pt = pytorch_results["cpu"].get(batch_key, {}).get("timings", {})
            pytorch_frontend_ms = (
                pt.get("fbank_ms", 0) + pt.get("patch_embed_ms", 0) +
                pt.get("layer_norm_ms", 0) + pt.get("proj_ms", 0)
            )
            if pytorch_frontend_ms > 0:
                speedup = pytorch_frontend_ms / onnx_ms
                print(f"{'Frontend':<20} {batch:<8} {onnx_ms:<12.2f} {pytorch_frontend_ms:<14.2f} {speedup:<10.2f}x")

        # Full encoder comparison
        if batch_key in encoder_bench:
            onnx_ms = encoder_bench[batch_key]["avg_ms"]
            pytorch_ms = pytorch_results["cpu"].get(batch_key, {}).get("timings", {}).get("total_ms", 0)
            if pytorch_ms > 0:
                speedup = pytorch_ms / onnx_ms
                print(f"{'Full Encoder':<20} {batch:<8} {onnx_ms:<12.2f} {pytorch_ms:<14.2f} {speedup:<10.2f}x")

print("\nDone!")
EOF

echo ""
echo "ONNX export complete!"
date
