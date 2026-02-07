"""Benchmark SSAMBA ONNX inference with mamba-onnx custom ops on CPU.

Exports each patch configuration to ONNX (with mamba::SelectiveScanFusedExact
custom ops) and benchmarks inference latency using ONNX Runtime on CPU.
This measures practical deployment cost for each token-context shape.

Uses the same 11 PATCH_CONFIGS as the GPU benchmark for direct comparison.

Usage:
    # Run single patch config (for SLURM array jobs):
    python benchmark_onnx_token_shapes.py --config-index 0 --results-dir onnx-results/

    # Run all configs sequentially:
    python benchmark_onnx_token_shapes.py --output onnx-results.json

    # Collect array job results:
    python benchmark_onnx_token_shapes.py --collect --results-dir onnx-results/

    # List configs:
    python benchmark_onnx_token_shapes.py --list-configs
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add ssamba_hf to path
ssamba_hf_dir = Path(__file__).resolve().parents[2] / "004-ssamba-hf-port" / "ssamba_hf"
sys.path.insert(0, str(ssamba_hf_dir))

import torch
from ssamba_hf import SSAMBAConfig, SSAMBAModel

from ssamba_onnx_export import export_ssamba

# -- Patch configurations (same as GPU benchmark) --
PATCH_CONFIGS = [
    ("coarse-128x24",      (128, 24), "Full-band 240ms"),
    ("medium-128x6",       (128, 6),  "Full-band 60ms"),
    ("wideframe-128x4",    (128, 4),  "Full-band 40ms"),
    ("baseline-16x16",     16,        "Square 16x16 (standard)"),
    ("fullband-128x2",     (128, 2),  "Full-band 20ms"),
    ("quarterband-32x4",   (32, 4),   "Quarter-band 40ms"),
    ("framelevel-128x1",   (128, 1),  "Frame-level 10ms"),
    ("halfframe-64x1",     (64, 1),   "Half-band 10ms"),
    ("quarterframe-32x1",  (32, 1),   "Quarter-band 10ms"),
    ("eighthframe-8x1",    (8, 1),    "Eighth-band 10ms"),
    ("fine-4x1",           (4, 1),    "32-band 10ms"),
]

MODEL_SIZES = [
    ("tiny", 192),
    ("small", 384),
]

MEL_BINS = 128
FPS = 100
DURATION_S = 60  # ONNX benchmark focuses on 60s (deployment-relevant)

# Token count thresholds for adaptive benchmarking
HIGH_TOKEN_THRESHOLD = 10_000
EXTREME_TOKEN_THRESHOLD = 30_000


def compute_expected_tokens(mel_bins, time_frames, patch_size):
    if isinstance(patch_size, int):
        fpatch, tpatch = patch_size, patch_size
    else:
        fpatch, tpatch = patch_size
    return (mel_bins // fpatch) * (time_frames // tpatch)


def benchmark_onnx_config(
    patch_name: str,
    patch_size,
    embed_dim: int,
    size_name: str,
    duration_s: int = DURATION_S,
    num_warmup: int = 3,
    num_runs: int = 10,
    ort_threads: int = 0,
) -> dict:
    """Export model to ONNX and benchmark with ONNX Runtime on CPU."""
    import onnxruntime as ort
    import mamba_onnx

    time_frames = duration_s * FPS
    expected_tokens = compute_expected_tokens(MEL_BINS, time_frames, patch_size)

    # Adaptive warmup/runs
    if expected_tokens >= EXTREME_TOKEN_THRESHOLD:
        num_warmup = 1
        num_runs = 3
    elif expected_tokens >= HIGH_TOKEN_THRESHOLD:
        num_warmup = 2
        num_runs = 5

    result = {
        "patch_name": patch_name,
        "patch_size": patch_size if isinstance(patch_size, int) else list(patch_size),
        "model_size": size_name,
        "embed_dim": embed_dim,
        "duration_s": duration_s,
        "time_frames": time_frames,
        "expected_tokens": expected_tokens,
    }

    try:
        # Build PyTorch model (CPU, no CUDA kernels)
        config = SSAMBAConfig(
            embed_dim=embed_dim,
            depth=24,
            img_size=(MEL_BINS, time_frames),
            patch_size=patch_size,
            use_cuda_kernels=False,
        )
        assert config.num_patches == expected_tokens

        model = SSAMBAModel(config)
        model.eval()
        result["parameters"] = sum(p.numel() for p in model.parameters())

        # Export to ONNX
        dummy_input = torch.randn(1, time_frames, MEL_BINS)
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")

            export_ssamba(model, dummy_input, onnx_path)
            onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            result["onnx_size_mb"] = round(onnx_size_mb, 1)

            # Load in ONNX Runtime with mamba-onnx custom ops
            sess_options = ort.SessionOptions()
            if ort_threads > 0:
                sess_options.intra_op_num_threads = ort_threads
                sess_options.inter_op_num_threads = ort_threads
            lib_path = mamba_onnx.register_custom_ops(sess_options)
            result["mamba_onnx_backend"] = Path(lib_path).stem

            session = ort.InferenceSession(
                onnx_path, sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Prepare input
            input_np = np.random.randn(1, time_frames, MEL_BINS).astype(np.float32)
            feed = {"input": input_np}

            # Warmup
            for _ in range(num_warmup):
                session.run(None, feed)

            # Timed runs
            times = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                session.run(None, feed)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

        result["mean_ms"] = round(mean_ms, 2)
        result["std_ms"] = round(std_ms, 2)
        result["tokens_per_sec"] = round(expected_tokens / (mean_ms / 1000), 1)
        result["num_runs"] = num_runs
        result["status"] = "ok"

        del model

    except Exception as e:
        result["status"] = f"error: {e}"
        result["mean_ms"] = None
        result["std_ms"] = None
        result["tokens_per_sec"] = None
        result["onnx_size_mb"] = None
        result["parameters"] = result.get("parameters")
        result["num_runs"] = 0

    return result


def run_single_config(config_index: int, results_dir: str, ort_threads: int = 0):
    """Run all model sizes for a single patch config (array task)."""
    if config_index < 0 or config_index >= len(PATCH_CONFIGS):
        print(f"ERROR: config-index {config_index} out of range [0, {len(PATCH_CONFIGS) - 1}]")
        sys.exit(1)

    patch_name, patch_size, desc = PATCH_CONFIGS[config_index]
    print(f"=== ONNX Benchmark — Config {config_index}: {patch_name} ({desc}) ===")
    print()

    results = []
    total = len(MODEL_SIZES)
    for idx, (size_name, embed_dim) in enumerate(MODEL_SIZES, 1):
        expected_tokens = compute_expected_tokens(MEL_BINS, DURATION_S * FPS, patch_size)
        print(
            f"[{idx}/{total}] {size_name} | {DURATION_S}s | "
            f"~{expected_tokens:,} tokens ... ",
            end="", flush=True,
        )

        r = benchmark_onnx_config(
            patch_name=patch_name,
            patch_size=patch_size,
            embed_dim=embed_dim,
            size_name=size_name,
            ort_threads=ort_threads,
        )
        results.append(r)

        if r["status"] == "ok":
            print(f"ok ({r['mean_ms']:.0f}ms, {r['onnx_size_mb']:.1f}MB onnx)")
        else:
            print(r["status"])

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    output_file = results_path / f"onnx-config-{config_index:02d}-{patch_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def run_all_configs(output_path: str, ort_threads: int = 0):
    """Run all configs sequentially."""
    total = len(MODEL_SIZES) * len(PATCH_CONFIGS)
    print(f"Running {total} ONNX benchmarks: {len(MODEL_SIZES)} model(s) x "
          f"{len(PATCH_CONFIGS)} patch config(s) @ {DURATION_S}s")
    print()

    results = []
    run_idx = 0

    for size_name, embed_dim in MODEL_SIZES:
        for patch_name, patch_size, desc in PATCH_CONFIGS:
            run_idx += 1
            expected_tokens = compute_expected_tokens(MEL_BINS, DURATION_S * FPS, patch_size)
            print(
                f"[{run_idx}/{total}] {patch_name} | {size_name} | "
                f"~{expected_tokens:,} tokens ... ",
                end="", flush=True,
            )

            r = benchmark_onnx_config(
                patch_name=patch_name,
                patch_size=patch_size,
                embed_dim=embed_dim,
                size_name=size_name,
                ort_threads=ort_threads,
            )
            results.append(r)

            if r["status"] == "ok":
                print(f"ok ({r['mean_ms']:.0f}ms)")
            else:
                print(r["status"])

    print_report(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def collect_results(results_dir: str, output_path: str):
    """Collect per-config JSON files into a combined report."""
    results_path = Path(results_dir)
    all_results = []

    json_files = sorted(results_path.glob("onnx-config-*.json"))
    if not json_files:
        print(f"ERROR: No onnx-config-*.json files found in {results_dir}")
        sys.exit(1)

    print(f"Collecting results from {len(json_files)} files in {results_dir}/")
    for f in json_files:
        with open(f) as fp:
            data = json.load(fp)
        all_results.extend(data)
        print(f"  {f.name}: {len(data)} results")

    print(f"\nTotal: {len(all_results)} benchmark results")
    print_report(all_results)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {output_path}")


def print_report(results: list):
    """Print formatted benchmark report."""
    print_table(results)
    print_token_scaling_summary(results)


def print_table(results: list):
    """Print results as a formatted table."""
    header = (
        f"{'Patch':<22} {'Model':<6} {'Tokens':>8} "
        f"{'Params':>10} {'ONNX(MB)':>9} {'ms/fwd':>9} {'tok/s':>10} {'Status':<8}"
    )
    print()
    print(header)
    print("-" * len(header))

    for size_name in ["tiny", "small"]:
        size_results = [r for r in results if r["model_size"] == size_name]
        if not size_results:
            continue
        size_results.sort(key=lambda r: r["expected_tokens"])

        for r in size_results:
            tokens = r.get("expected_tokens", "?")
            params = r.get("parameters")
            params_str = f"{params:,}" if params else "N/A"
            onnx_mb = r.get("onnx_size_mb")
            onnx_str = f"{onnx_mb:.1f}" if onnx_mb is not None else "N/A"
            ms = r.get("mean_ms")
            ms_str = f"{ms:.1f}" if ms is not None else "N/A"
            tps = r.get("tokens_per_sec")
            tps_str = f"{tps:,.1f}" if tps is not None else "N/A"

            print(
                f"{r['patch_name']:<22} {r['model_size']:<6} "
                f"{tokens:>8} {params_str:>10} {onnx_str:>9} {ms_str:>9} "
                f"{tps_str:>10} {r['status']:<8}"
            )
        if size_name == "tiny":
            print()

    print()


def print_token_scaling_summary(results: list):
    """Print how latency scales with token count."""
    print("=== Token Count vs ONNX Latency ===")
    for size_name in ["tiny", "small"]:
        size_results = [r for r in results if r["model_size"] == size_name]
        if not size_results:
            continue
        print(f"\n  {size_name} ({DURATION_S}s):")
        ok_results = [r for r in size_results if r["status"] == "ok"]
        ok_results.sort(key=lambda r: r["expected_tokens"])
        for r in ok_results:
            bar_len = min(50, max(1, int(r["mean_ms"] / 500)))
            bar = "#" * bar_len
            print(
                f"    {r['expected_tokens']:>8} tok | {r['mean_ms']:>9.1f} ms | "
                f"{r.get('onnx_size_mb', 0) or 0:>6.1f} MB | {bar}"
            )
        fail_results = [r for r in size_results if r["status"] != "ok"]
        fail_results.sort(key=lambda r: r["expected_tokens"])
        for r in fail_results:
            print(f"    {r['expected_tokens']:>8} tok |  {r['status']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SSAMBA ONNX inference with mamba-onnx custom ops"
    )
    parser.add_argument("--output", type=str, default=None, help="Path for combined results JSON")
    parser.add_argument(
        "--config-index", type=int, default=None,
        help=f"Run only this patch config index [0-{len(PATCH_CONFIGS)-1}]",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory for per-config results or collection source",
    )
    parser.add_argument("--collect", action="store_true", help="Collect results from --results-dir")
    parser.add_argument("--list-configs", action="store_true", help="List all patch configs")
    parser.add_argument(
        "--ort-threads", type=int, default=0,
        help="Number of ORT intra/inter-op threads (0=default/all)",
    )

    args = parser.parse_args()

    if args.list_configs:
        time_frames = DURATION_S * FPS
        print(f"{'Index':<6} {'Name':<22} {'Patch':<12} {'Tok@{DURATION_S}s':>10}  {'Description'}")
        print("-" * 70)
        for i, (name, ps, desc) in enumerate(PATCH_CONFIGS):
            tok = compute_expected_tokens(MEL_BINS, time_frames, ps)
            ps_str = f"{ps}" if isinstance(ps, int) else f"{ps[0]}x{ps[1]}"
            print(f"{i:<6} {name:<22} {ps_str:<12} {tok:>10,}  {desc}")
        print(f"\nTotal configs: {len(PATCH_CONFIGS)}")
        return

    if args.collect:
        results_dir = args.results_dir or str(
            Path(__file__).resolve().parent.parent / "onnx-results"
        )
        output_path = args.output or str(
            Path(__file__).resolve().parent.parent / "onnx-results.json"
        )
        collect_results(results_dir, output_path)
        return

    # Verify mamba-onnx is available
    try:
        import mamba_onnx
        backends = mamba_onnx.available_backends()
        print(f"mamba-onnx backends: {backends}")
        if not backends:
            print("ERROR: mamba-onnx installed but no backends found (missing .so files)")
            sys.exit(1)
    except ImportError:
        print("ERROR: mamba-onnx not installed. Install with:")
        print('  uv pip install "mamba-onnx @ https://github.com/acmiyaguchi/mamba-onnx/releases/latest/download/mamba_onnx-0.1.0-py3-none-manylinux_2_17_x86_64.whl"')
        sys.exit(1)

    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"ORT threads: {args.ort_threads if args.ort_threads > 0 else 'default'}")
    print()

    if args.config_index is not None:
        results_dir = args.results_dir or str(
            Path(__file__).resolve().parent.parent / "onnx-results"
        )
        run_single_config(args.config_index, results_dir, args.ort_threads)
        return

    output_path = args.output or str(
        Path(__file__).resolve().parent.parent / "onnx-results.json"
    )
    run_all_configs(output_path, args.ort_threads)


if __name__ == "__main__":
    main()
