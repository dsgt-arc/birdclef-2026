#!/bin/bash
#SBATCH --job-name=ssamba-profile
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/ssamba-profile-%j.out
set -eu

MODEL_SIZE=${MODEL_SIZE:-small}
USE_CUDA_KERNELS=${USE_CUDA_KERNELS:-0}

echo "Profiling SSAMBA-HF-$MODEL_SIZE inference"
echo "CUDA kernels: $USE_CUDA_KERNELS"
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

ckpt_path=$scratch_dir/checkpoints/ssamba/ssamba_${MODEL_SIZE}_400.pth
if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: SSAMBA checkpoint not found at $ckpt_path"
    exit 1
fi

# Use copy mode to avoid hardlink failures between scratch and tmpdir filesystems
export UV_LINK_MODE=copy
export HF_HOME=$scratch_dir/.cache/huggingface

# Install with or without CUDA kernels
cd $repo_dir/user/anthony
if [ "$USE_CUDA_KERNELS" = "1" ]; then
    echo "Installing with CUDA kernel support..."
    # Install to local NVMe for CUDA kernel builds
    export UV_PROJECT_ENVIRONMENT=${TMPDIR}/${SLURM_JOB_ID}/.venv
    uv sync --extra cuda

    # Diagnostic: test mamba-ssm CUDA kernel import
    echo ""
    echo "=== Testing mamba-ssm CUDA kernels ==="
    ${UV_PROJECT_ENVIRONMENT}/bin/python -c "
import sys
print(f'Python: {sys.executable}')
try:
    import mamba_ssm
    print(f'mamba_ssm version: {mamba_ssm.__version__}')
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print('CUDA kernels: AVAILABLE')
except Exception as e:
    print(f'mamba_ssm import failed: {e}')
    sys.exit(1)
"
    echo "=== End mamba-ssm test ==="
    echo ""
else
    export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
    uv sync
fi

cd $repo_dir

${UV_PROJECT_ENVIRONMENT}/bin/python << PYEOF
"""SSAMBA-HF inference profiling: stage-level timing + batch scaling."""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Add ssamba_hf to path
ssamba_hf_dir = Path("user/anthony/experiments/004-ssamba-hf-port/ssamba_hf")
sys.path.insert(0, str(ssamba_hf_dir))
from ssamba_hf import load_from_ssamba_checkpoint

# ----- Config -----
CHECKPOINT = "$ckpt_path"
MODEL_SIZE = "$MODEL_SIZE"
DEVICE_GPU = "cuda" if torch.cuda.is_available() else None
DEVICE_CPU = "cpu"
NUM_SAMPLES_5S = 80000   # 5 seconds at 16kHz
NUM_SAMPLES_10S = 160000  # 10 seconds at 16kHz
GPU_BATCH_SIZES = [1, 4, 16, 32]
CPU_BATCH_SIZES = [1, 4, 16]
WARMUP_ITERS = 5
BENCH_ITERS = 20
CPU_BENCH_ITERS = 10

# ----- Helper functions -----

def sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

def reset_mem(device):
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_mem(device):
    if device == "cuda" and torch.cuda.is_available():
        return {
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "allocated_gpu_mb": torch.cuda.memory_allocated() / (1024**2),
        }
    return {}

def waveform_to_fbank(waveform, target_tdim=1024, target_fdim=128, norm_mean=-4.2677393, norm_std=4.5689974):
    """Convert waveform to fbank features."""
    batch_size = waveform.shape[0]
    fbanks = []
    for i in range(batch_size):
        wav = waveform[i] - waveform[i].mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=target_fdim,
            dither=0.0,
            frame_shift=10,
        )
        n_frames = fbank.shape[0]
        if n_frames < target_tdim:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_tdim - n_frames))
        elif n_frames > target_tdim:
            fbank = fbank[:target_tdim, :]
        fbank = (fbank - norm_mean) / (norm_std * 2)
        fbanks.append(fbank)
    return torch.stack(fbanks, dim=0)


def profile_forward(model, fbank, device):
    """Profile forward pass with stage-level timing.

    Returns dict with per-stage timings in ms.
    """
    timings = {}

    # --- reshape ---
    x = fbank.unsqueeze(1).transpose(2, 3).to(device)  # (B, 1, F, T)
    B = x.shape[0]

    sync(device)
    t0 = time.perf_counter()

    # Stage 1: patch embedding
    sync(device)
    s = time.perf_counter()
    x_patched = model.patch_embed(x)
    sync(device)
    timings["patch_embed_ms"] = (time.perf_counter() - s) * 1000

    num_patches = x_patched.shape[1]

    # Stage 2: CLS token + positional embedding
    sync(device)
    s = time.perf_counter()
    if model.cls_token is not None:
        cls_tokens = model.cls_token.expand(B, -1, -1)
        if model.config.use_middle_cls_token:
            pos = num_patches // 2
            x_patched = torch.cat([x_patched[:, :pos, :], cls_tokens, x_patched[:, pos:, :]], dim=1)
        else:
            x_patched = torch.cat([cls_tokens, x_patched], dim=1)
    x_patched = x_patched + model.pos_embed
    x_patched = model.pos_drop(x_patched)
    sync(device)
    timings["pos_embed_ms"] = (time.perf_counter() - s) * 1000

    # Stage 3: Mamba layers (bidirectional)
    sync(device)
    s = time.perf_counter()
    hidden_states, residual = model.layers(x_patched)
    sync(device)
    timings["mamba_layers_ms"] = (time.perf_counter() - s) * 1000

    # Stage 4: Final norm
    sync(device)
    s = time.perf_counter()
    if residual is None:
        residual = hidden_states
    else:
        residual = residual + model.drop_path(hidden_states)
    hidden_states = model.norm_f(residual.to(dtype=model.norm_f.weight.dtype))
    x_out = model.norm_f(hidden_states)  # double norm (matches original)
    sync(device)
    timings["norm_ms"] = (time.perf_counter() - s) * 1000

    # Stage 5: Pooling
    sync(device)
    s = time.perf_counter()
    if model.cls_token is not None and model.config.use_middle_cls_token:
        cls_pos = num_patches // 2
        tokens_before = x_out[:, :cls_pos, :]
        tokens_after = x_out[:, cls_pos + 1:, :]
        result = torch.cat([tokens_before, tokens_after], dim=1).mean(dim=1)
    else:
        result = x_out[:, 1:, :].mean(dim=1) if model.cls_token is not None else x_out.mean(dim=1)
    sync(device)
    timings["pool_ms"] = (time.perf_counter() - s) * 1000

    sync(device)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return result, timings


def run_benchmark(model, device, batch_sizes, num_samples, warmup, iters, label):
    """Run benchmark across batch sizes."""
    print(f"\n{'='*60}")
    print(f"{label} BENCHMARKS")
    print(f"{'='*60}")

    if device == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(4)
        print(f"CPU threads: {torch.get_num_threads()}")

    # Load model
    model_dev = model.to(device)
    model_dev.eval()

    # Determine target tdim from model config
    target_tdim = model.config.img_size[1]
    target_fdim = model.config.img_size[0]

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        reset_mem(device)

        # Create synthetic audio and convert to fbank on CPU
        audio = torch.randn(batch_size, num_samples)
        fbank = waveform_to_fbank(audio, target_tdim=target_tdim, target_fdim=target_fdim)

        # Time fbank separately (always on CPU)
        fbank_times = []
        for _ in range(min(3, iters)):
            s = time.perf_counter()
            _ = waveform_to_fbank(audio, target_tdim=target_tdim, target_fdim=target_fdim)
            fbank_times.append((time.perf_counter() - s) * 1000)
        fbank_ms = sum(fbank_times) / len(fbank_times)

        # Warmup model forward
        with torch.no_grad():
            for _ in range(warmup):
                _ = profile_forward(model_dev, fbank, device)

        # Benchmark model forward
        all_timings = []
        with torch.no_grad():
            for _ in range(iters):
                _, timings = profile_forward(model_dev, fbank, device)
                all_timings.append(timings)

        mem = get_mem(device)

        # Average timings
        avg = {}
        for key in all_timings[0]:
            avg[key] = sum(t[key] for t in all_timings) / len(all_timings)

        # Add fbank time
        avg["fbank_ms"] = fbank_ms
        avg["total_with_fbank_ms"] = avg["total_ms"] + fbank_ms

        throughput = (batch_size * 1000) / avg["total_with_fbank_ms"] if avg["total_with_fbank_ms"] > 0 else 0

        result = {
            "batch_size": batch_size,
            "num_audio_samples": num_samples,
            "timings": avg,
            "memory": mem,
            "throughput_samples_per_sec": throughput,
        }
        results[f"batch_{batch_size}"] = result

        # Print results
        print(f"  Total (model only): {avg['total_ms']:.2f} ms")
        print(f"  Total (with fbank): {avg['total_with_fbank_ms']:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Breakdown:")
        for stage in ["fbank_ms", "patch_embed_ms", "pos_embed_ms", "mamba_layers_ms", "norm_ms", "pool_ms"]:
            pct = 100 * avg[stage] / avg["total_with_fbank_ms"] if avg["total_with_fbank_ms"] > 0 else 0
            print(f"    {stage:20s}: {avg[stage]:8.2f} ms ({pct:5.1f}%)")
        if mem:
            print(f"  Peak GPU: {mem.get('peak_gpu_mb', 0):.1f} MB")

    return results


# ===== Main =====
print(f"Loading SSAMBA-HF-{MODEL_SIZE} from {CHECKPOINT}")
model, config = load_from_ssamba_checkpoint(CHECKPOINT, device="cpu", verbose=True)
model.eval()
print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Embed dim: {config.embed_dim}")
print(f"Depth: {config.depth}")
print(f"Input size: {config.img_size}")

# Check for CUDA kernels
has_cuda_kernels = False
try:
    import mamba_ssm
    has_cuda_kernels = True
    print(f"mamba-ssm version: {mamba_ssm.__version__} (CUDA kernels available)")
except ImportError:
    print("mamba-ssm not installed (using PyTorch fallback)")

all_results = {
    "model": {
        "name": f"ssamba-hf-{MODEL_SIZE}",
        "params": sum(p.numel() for p in model.parameters()),
        "embed_dim": config.embed_dim,
        "depth": config.depth,
        "img_size": list(config.img_size),
        "cuda_kernels": has_cuda_kernels,
    },
    "config": {
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
        "cpu_bench_iters": CPU_BENCH_ITERS,
        "audio_samples_5s": NUM_SAMPLES_5S,
    },
}

# GPU benchmarks
if DEVICE_GPU:
    model_gpu = model.to(DEVICE_GPU)
    gpu_results = run_benchmark(
        model_gpu, DEVICE_GPU, GPU_BATCH_SIZES, NUM_SAMPLES_5S,
        WARMUP_ITERS, BENCH_ITERS, "GPU"
    )
    all_results["gpu"] = gpu_results
    del model_gpu
    torch.cuda.empty_cache()

# CPU benchmarks
model_cpu = model.to(DEVICE_CPU)
cpu_results = run_benchmark(
    model_cpu, DEVICE_CPU, CPU_BATCH_SIZES, NUM_SAMPLES_5S,
    3, CPU_BENCH_ITERS, "CPU"
)
all_results["cpu"] = cpu_results

# ===== Comparison summary =====
print(f"\n{'='*60}")
print("SUMMARY: SSAMBA-HF vs BEATs (from experiment 002)")
print(f"{'='*60}")
print(f"{'Device':<8} {'Batch':<8} {'Total (ms)':<14} {'Throughput':<16} {'Mamba/Xfmr %':<14}")
print("-" * 60)

# BEATs reference numbers from experiment 002 (L40S GPU, 4-thread CPU)
beats_ref = {
    "gpu": {1: (8.0, 125), 4: (12.5, 320), 16: (43.6, 367), 32: (91.8, 349)},
    "cpu": {1: (124, 8.1), 4: (447, 8.9), 16: (1852, 8.6)},
}

for device_key in ["gpu", "cpu"]:
    if device_key not in all_results:
        continue
    for key, val in all_results[device_key].items():
        if not key.startswith("batch_"):
            continue
        batch = val["batch_size"]
        total = val["timings"]["total_with_fbank_ms"]
        tput = val["throughput_samples_per_sec"]
        mamba_pct = 100 * val["timings"]["mamba_layers_ms"] / total if total > 0 else 0
        label = device_key.upper()

        # BEATs comparison
        beats_total, beats_tput = beats_ref.get(device_key, {}).get(batch, (0, 0))
        speedup = beats_total / total if total > 0 and beats_total > 0 else 0

        line = f"{label:<8} {batch:<8} {total:<14.2f} {tput:<16.1f} {mamba_pct:<14.1f}"
        if speedup > 0:
            line += f"  (vs BEATs: {speedup:.2f}x)"
        print(line)

# Save results
output_dir = Path("$scratch_dir/outputs/exp001-ssamba-profile")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"ssamba_hf_{MODEL_SIZE}_profile.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")
print("\nDone!")
PYEOF

echo ""
echo "Profiling complete!"
date
