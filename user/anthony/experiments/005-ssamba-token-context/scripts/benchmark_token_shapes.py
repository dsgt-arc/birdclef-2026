"""Benchmark SSAMBA with different patch shapes and audio durations.

Tests rectangular patch configurations spanning ~100 to ~100k tokens to map out
the practical limits of Mamba's O(n) scaling. No training -- synthetic data only.

Patch configs are designed to cover:
1. Orders of magnitude in token count (~125 to ~96k at 30s)
2. Temporal vs frequency resolution tradeoffs (full-band vs freq-split at same token count)
3. O(n) scaling verification via 30s vs 60s comparison

Usage:
    # Run single patch config (for SLURM array jobs):
    python benchmark_token_shapes.py --config-index 0 --results-dir results/

    # Run all configs sequentially:
    python benchmark_token_shapes.py --output results.json

    # Collect array job results into combined report:
    python benchmark_token_shapes.py --collect --results-dir results/

    # List configs and their indices:
    python benchmark_token_shapes.py --list-configs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add ssamba_hf to path
ssamba_hf_dir = Path(__file__).resolve().parents[2] / "004-ssamba-hf-port" / "ssamba_hf"
sys.path.insert(0, str(ssamba_hf_dir))

from ssamba_hf import SSAMBAConfig, SSAMBAModel


# -- Patch configurations --
# Each: (name, patch_size, description)
# Ordered by approximate token count at 30s (3000 frames, 128 mel bins)
PATCH_CONFIGS = [
    # ~125 tokens @30s — very coarse, sentence-level
    ("coarse-128x24",      (128, 24), "Full-band 240ms"),
    # ~500 tokens @30s — coarse, word-level
    ("medium-128x6",       (128, 6),  "Full-band 60ms"),
    # ~750 tokens @30s — wide frame
    ("wideframe-128x4",    (128, 4),  "Full-band 40ms"),
    # ~1500 tokens @30s — two configs at same count, different shapes
    ("baseline-16x16",     16,        "Square 16x16 (standard)"),
    ("fullband-128x2",     (128, 2),  "Full-band 20ms"),
    # ~3000 tokens @30s — three configs: temporal vs frequency tradeoff
    ("quarterband-32x4",   (32, 4),   "Quarter-band 40ms"),
    ("framelevel-128x1",   (128, 1),  "Frame-level 10ms"),
    # ~6000 tokens @30s
    ("halfframe-64x1",     (64, 1),   "Half-band 10ms"),
    # ~12000 tokens @30s
    ("quarterframe-32x1",  (32, 1),   "Quarter-band 10ms"),
    # ~48000 tokens @30s — likely OOM for small model
    ("eighthframe-8x1",    (8, 1),    "Eighth-band 10ms"),
    # ~96000 tokens @30s — stress test
    ("fine-4x1",           (4, 1),    "32-band 10ms"),
]

# Model sizes: (name, embed_dim)
MODEL_SIZES = [
    ("tiny", 192),
    ("small", 384),
]

# Audio durations in seconds (at 100 fps -> time_frames)
DURATIONS = [30, 60]

MEL_BINS = 128
FPS = 100  # frames per second

# Token count thresholds for adaptive benchmarking
HIGH_TOKEN_THRESHOLD = 10_000   # reduce warmup/runs above this
EXTREME_TOKEN_THRESHOLD = 30_000  # minimal warmup/runs above this


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def compute_expected_tokens(mel_bins, time_frames, patch_size):
    """Compute expected token count for a given configuration."""
    if isinstance(patch_size, int):
        fpatch, tpatch = patch_size, patch_size
    else:
        fpatch, tpatch = patch_size
    f_patches = mel_bins // fpatch
    t_patches = time_frames // tpatch
    return f_patches * t_patches


def benchmark_config(
    patch_name: str,
    patch_size,
    embed_dim: int,
    size_name: str,
    duration_s: int,
    device: torch.device,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark a single configuration. Returns results dict or error info."""
    time_frames = duration_s * FPS
    expected_tokens = compute_expected_tokens(MEL_BINS, time_frames, patch_size)

    # Adaptive warmup/runs based on token count
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
        config = SSAMBAConfig(
            embed_dim=embed_dim,
            depth=24,
            img_size=(MEL_BINS, time_frames),
            patch_size=patch_size,
        )

        # Verify token count
        assert config.num_patches == expected_tokens, (
            f"num_patches mismatch: config={config.num_patches}, expected={expected_tokens}"
        )

        model = SSAMBAModel(config).to(device)
        model.eval()

        params = count_parameters(model)
        result["parameters"] = params

        # Synthetic input: (batch, time_frames, mel_bins)
        x = torch.randn(1, time_frames, MEL_BINS, device=device)

        # Reset memory stats
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Reset after warmup to measure steady-state memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

        result["mean_ms"] = round(mean_ms, 2)
        result["std_ms"] = round(std_ms, 2)
        result["tokens_per_sec"] = round(expected_tokens / (mean_ms / 1000), 1)
        result["num_runs"] = num_runs

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device)
            result["peak_gpu_mb"] = round(peak_mem / 1024 / 1024, 1)
        else:
            result["peak_gpu_mb"] = None

        result["status"] = "ok"

        # Cleanup
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        result["mean_ms"] = None
        result["std_ms"] = None
        result["tokens_per_sec"] = None
        result["peak_gpu_mb"] = None
        result["parameters"] = None
        result["num_runs"] = 0
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = f"error: {e}"
        result["mean_ms"] = None
        result["std_ms"] = None
        result["tokens_per_sec"] = None
        result["peak_gpu_mb"] = None
        result["parameters"] = None
        result["num_runs"] = 0

    return result


def run_single_config(config_index: int, results_dir: str, device: torch.device):
    """Run all model sizes and durations for a single patch config (array task)."""
    if config_index < 0 or config_index >= len(PATCH_CONFIGS):
        print(f"ERROR: config-index {config_index} out of range [0, {len(PATCH_CONFIGS) - 1}]")
        sys.exit(1)

    patch_name, patch_size, desc = PATCH_CONFIGS[config_index]
    print(f"=== Config {config_index}: {patch_name} ({desc}) ===")
    print()

    results = []
    total = len(MODEL_SIZES) * len(DURATIONS)
    run_idx = 0

    for size_name, embed_dim in MODEL_SIZES:
        for duration_s in DURATIONS:
            run_idx += 1
            expected_tokens = compute_expected_tokens(MEL_BINS, duration_s * FPS, patch_size)
            print(
                f"[{run_idx}/{total}] {size_name} | {duration_s}s | "
                f"~{expected_tokens:,} tokens ... ",
                end="", flush=True,
            )

            r = benchmark_config(
                patch_name=patch_name,
                patch_size=patch_size,
                embed_dim=embed_dim,
                size_name=size_name,
                duration_s=duration_s,
                device=device,
            )
            results.append(r)

            if r["status"] == "ok":
                print(f"ok ({r['mean_ms']:.0f}ms, {r.get('peak_gpu_mb', '?')}MB)")
            else:
                print(r["status"])

    # Save per-config results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    output_file = results_path / f"config-{config_index:02d}-{patch_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


def run_all_configs(device: torch.device, output_path: str):
    """Run all configs sequentially (non-array mode)."""
    total_runs = len(MODEL_SIZES) * len(PATCH_CONFIGS) * len(DURATIONS)
    print(f"Running {total_runs} benchmarks: {len(MODEL_SIZES)} model(s) x "
          f"{len(PATCH_CONFIGS)} patch config(s) x {len(DURATIONS)} duration(s)")
    print()

    results = []
    run_idx = 0

    for size_name, embed_dim in MODEL_SIZES:
        for patch_name, patch_size, desc in PATCH_CONFIGS:
            for duration_s in DURATIONS:
                run_idx += 1
                expected_tokens = compute_expected_tokens(MEL_BINS, duration_s * FPS, patch_size)
                print(
                    f"[{run_idx}/{total_runs}] {patch_name} | {size_name} | "
                    f"{duration_s}s | ~{expected_tokens:,} tokens ... ",
                    end="", flush=True,
                )

                r = benchmark_config(
                    patch_name=patch_name,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    size_name=size_name,
                    duration_s=duration_s,
                    device=device,
                )
                results.append(r)

                if r["status"] == "ok":
                    print(f"ok ({r['mean_ms']:.0f}ms, {r.get('peak_gpu_mb', '?')}MB)")
                else:
                    print(r["status"])

    print_report(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def collect_results(results_dir: str, output_path: str):
    """Collect per-config JSON files from array jobs into a combined report."""
    results_path = Path(results_dir)
    all_results = []

    json_files = sorted(results_path.glob("config-*.json"))
    if not json_files:
        print(f"ERROR: No config-*.json files found in {results_dir}")
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


def print_report(results: list[dict]):
    """Print full benchmark report (table + scaling checks)."""
    print_table(results)
    check_linear_scaling(results)
    print_token_scaling_summary(results)


def print_table(results: list[dict]):
    """Print results as a formatted table, sorted by token count within each model size."""
    header = (
        f"{'Patch':<22} {'Model':<6} {'Dur':>4} {'Tokens':>8} "
        f"{'Params':>10} {'Mem(MB)':>8} {'ms/fwd':>9} {'tok/s':>10} {'Status':<8}"
    )
    print()
    print(header)
    print("-" * len(header))

    # Group by model size
    for size_name in ["tiny", "small"]:
        size_results = [r for r in results if r["model_size"] == size_name]
        if not size_results:
            continue
        # Sort by token count then duration
        size_results.sort(key=lambda r: (r["expected_tokens"], r["duration_s"]))

        for r in size_results:
            tokens = r.get("expected_tokens", "?")
            params = r.get("parameters")
            params_str = f"{params:,}" if params else "N/A"
            mem = r.get("peak_gpu_mb")
            mem_str = f"{mem:,.1f}" if mem is not None else "N/A"
            ms = r.get("mean_ms")
            ms_str = f"{ms:.1f}" if ms is not None else "N/A"
            tps = r.get("tokens_per_sec")
            tps_str = f"{tps:,.1f}" if tps is not None else "N/A"

            print(
                f"{r['patch_name']:<22} {r['model_size']:<6} {r['duration_s']:>3}s "
                f"{tokens:>8} {params_str:>10} {mem_str:>8} {ms_str:>9} "
                f"{tps_str:>10} {r['status']:<8}"
            )
        if size_name == "tiny":
            print()  # separator between model sizes

    print()


def check_linear_scaling(results: list[dict]):
    """Check 60s/30s time ratio to verify O(n) scaling."""
    print("=== O(n) Scaling Check (60s/30s time ratio, expect ~2.0) ===")
    # Group by (patch_name, model_size)
    grouped = {}
    for r in results:
        key = (r["patch_name"], r["model_size"])
        grouped.setdefault(key, {})[r["duration_s"]] = r

    for (patch, size), durations in sorted(grouped.items()):
        if 30 in durations and 60 in durations:
            t30 = durations[30].get("mean_ms")
            t60 = durations[60].get("mean_ms")
            if t30 and t60:
                ratio = t60 / t30
                verdict = "LINEAR" if ratio < 2.5 else "SUPERLINEAR"
                print(f"  {patch:<22} ({size}): {t30:.1f}ms -> {t60:.1f}ms, ratio={ratio:.2f} [{verdict}]")
            else:
                s30 = durations[30].get("status", "?")
                s60 = durations[60].get("status", "?")
                print(f"  {patch:<22} ({size}): 30s={s30}, 60s={s60}")
    print()


def print_token_scaling_summary(results: list[dict]):
    """Print summary showing how throughput scales with token count."""
    print("=== Token Count Scaling Summary ===")
    for size_name in ["tiny", "small"]:
        size_30s = [r for r in results if r["model_size"] == size_name and r["duration_s"] == 30]
        if not size_30s:
            continue
        print(f"\n  {size_name} (30s):")
        ok_results = [r for r in size_30s if r["status"] == "ok"]
        ok_results.sort(key=lambda r: r["expected_tokens"])
        for r in ok_results:
            bar_len = min(50, max(1, int(r["mean_ms"] / 100)))
            bar = "#" * bar_len
            print(
                f"    {r['expected_tokens']:>8} tok | {r['mean_ms']:>9.1f} ms | "
                f"{r.get('peak_gpu_mb', 0) or 0:>7.0f} MB | {bar}"
            )
        # Show OOM/error configs
        fail_results = [r for r in size_30s if r["status"] != "ok"]
        fail_results.sort(key=lambda r: r["expected_tokens"])
        for r in fail_results:
            print(f"    {r['expected_tokens']:>8} tok |  {r['status']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark SSAMBA token shapes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no GPU memory/timing)")
    parser.add_argument("--output", type=str, default=None, help="Path for combined results JSON")

    # Array job mode
    parser.add_argument(
        "--config-index", type=int, default=None,
        help=f"Run only this patch config index [0-{len(PATCH_CONFIGS)-1}] (for SLURM array jobs)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory for per-config result files (array mode) or collection source (--collect)",
    )

    # Collection mode
    parser.add_argument(
        "--collect", action="store_true",
        help="Collect per-config results from --results-dir into a combined report",
    )

    # Utility
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List all patch configurations with their indices and exit",
    )

    args = parser.parse_args()

    # List configs mode
    if args.list_configs:
        print(f"{'Index':<6} {'Name':<22} {'Patch':<12} {'Tok@30s':>8} {'Tok@60s':>8}  {'Description'}")
        print("-" * 80)
        for i, (name, ps, desc) in enumerate(PATCH_CONFIGS):
            t30 = compute_expected_tokens(MEL_BINS, 30 * FPS, ps)
            t60 = compute_expected_tokens(MEL_BINS, 60 * FPS, ps)
            ps_str = f"{ps}" if isinstance(ps, int) else f"{ps[0]}x{ps[1]}"
            print(f"{i:<6} {name:<22} {ps_str:<12} {t30:>8,} {t60:>8,}  {desc}")
        print(f"\nTotal configs: {len(PATCH_CONFIGS)} (use --config-index N to run one)")
        return

    # Collect mode
    if args.collect:
        results_dir = args.results_dir or str(
            Path(__file__).resolve().parent.parent / "results"
        )
        output_path = args.output or str(
            Path(__file__).resolve().parent.parent / "results.json"
        )
        collect_results(results_dir, output_path)
        return

    # Benchmark mode — set up device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU (timing will be approximate, no memory tracking)")
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"Running on GPU: {gpu_name} ({gpu_mem:.0f} MB)")

    # Check CUDA kernel availability
    from ssamba_hf.mamba import is_cuda_kernel_available
    cuda_kernels = is_cuda_kernel_available()
    print(f"CUDA Mamba kernels: {'available' if cuda_kernels else 'NOT available (using pure PyTorch)'}")
    if not cuda_kernels and device.type != "cpu":
        print("WARNING: Without CUDA kernels, high token counts will be extremely slow")
    print()

    # Array job mode: run single config
    if args.config_index is not None:
        results_dir = args.results_dir or str(
            Path(__file__).resolve().parent.parent / "results"
        )
        run_single_config(args.config_index, results_dir, device)
        return

    # Default: run all configs sequentially
    output_path = args.output or str(
        Path(__file__).resolve().parent.parent / "results.json"
    )
    run_all_configs(device, output_path)


if __name__ == "__main__":
    main()
