"""Profiling utilities for BEATs inference benchmarking.

Provides stage-level timing and memory tracking for analyzing
where compute time is spent in the BEATs pipeline.
"""

import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add vendor path for BEATs imports
VENDOR_PATH = Path(__file__).parents[4] / "vendor" / "unilm" / "beats"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))


@dataclass
class StageTimings:
    """Timing results for each stage of BEATs inference."""

    fbank_ms: float = 0.0
    patch_embed_ms: float = 0.0
    layer_norm_ms: float = 0.0
    proj_ms: float = 0.0
    transformer_ms: float = 0.0
    pool_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fbank_ms": self.fbank_ms,
            "patch_embed_ms": self.patch_embed_ms,
            "layer_norm_ms": self.layer_norm_ms,
            "proj_ms": self.proj_ms,
            "transformer_ms": self.transformer_ms,
            "pool_ms": self.pool_ms,
            "total_ms": self.total_ms,
        }

    @property
    def frontend_ms(self) -> float:
        """Time for frontend (everything before transformer)."""
        return self.fbank_ms + self.patch_embed_ms + self.layer_norm_ms + self.proj_ms

    def breakdown_pct(self) -> dict:
        """Get percentage breakdown by stage."""
        if self.total_ms == 0:
            return {}
        return {
            "fbank_pct": 100 * self.fbank_ms / self.total_ms,
            "patch_embed_pct": 100 * self.patch_embed_ms / self.total_ms,
            "layer_norm_pct": 100 * self.layer_norm_ms / self.total_ms,
            "proj_pct": 100 * self.proj_ms / self.total_ms,
            "transformer_pct": 100 * self.transformer_ms / self.total_ms,
            "pool_pct": 100 * self.pool_ms / self.total_ms,
        }


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_gpu_mb: float = 0.0
    peak_ram_mb: float = 0.0
    allocated_gpu_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peak_gpu_mb": self.peak_gpu_mb,
            "peak_ram_mb": self.peak_ram_mb,
            "allocated_gpu_mb": self.allocated_gpu_mb,
        }


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    batch_size: int
    num_samples: int
    timings: StageTimings
    memory: MemoryStats
    throughput_samples_per_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "timings": self.timings.to_dict(),
            "timings_breakdown_pct": self.timings.breakdown_pct(),
            "memory": self.memory.to_dict(),
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
        }


class ProfiledBEATs(nn.Module):
    """BEATs model wrapper with stage-level profiling.

    Wraps the BEATs model to measure time spent in each stage:
    - fbank: mel-filterbank extraction
    - patch_embed: Conv2d patch embedding
    - layer_norm: LayerNorm on patch embeddings
    - proj: Linear projection to encoder dimension
    - transformer: 12-layer TransformerEncoder
    - pool: Mean pooling over patches

    Args:
        checkpoint_path: Path to BEATs checkpoint file
        device: Device to run on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(device)

        self.cfg = cfg
        self._embed_dim = cfg.encoder_embed_dim

        # Timing accumulators
        self._timings: list[StageTimings] = []

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _sync_cuda(self):
        """Synchronize CUDA for accurate timing."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _time_stage(self, fn, *args, **kwargs):
        """Time a function call with CUDA synchronization."""
        self._sync_cuda()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self._sync_cuda()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    @torch.no_grad()
    def forward_profiled(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> tuple[torch.Tensor, StageTimings]:
        """Forward pass with stage-level timing.

        Args:
            source: Audio waveform tensor of shape (batch, samples) at 16kHz
            fbank_mean: Mean for fbank normalization
            fbank_std: Std for fbank normalization

        Returns:
            Tuple of (embeddings, timings) where embeddings is (batch, embed_dim)
        """
        timings = StageTimings()
        total_start = time.perf_counter()

        # Stage 1: fbank extraction
        fbank, timings.fbank_ms = self._time_stage(
            self.model.preprocess, source, fbank_mean, fbank_std
        )

        # Stage 2a: patch embedding
        fbank = fbank.unsqueeze(1)

        def patch_embed_fn(x):
            features = self.model.patch_embedding(x)
            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.transpose(1, 2)
            return features

        features, timings.patch_embed_ms = self._time_stage(patch_embed_fn, fbank)

        # Stage 2b: layer norm
        features, timings.layer_norm_ms = self._time_stage(self.model.layer_norm, features)

        # Stage 2c: projection
        if self.model.post_extract_proj is not None:
            features, timings.proj_ms = self._time_stage(
                self.model.post_extract_proj, features
            )

        # Apply dropout (not timed, negligible)
        x = self.model.dropout_input(features)

        # Stage 3: transformer encoder
        def transformer_fn(x):
            x, _ = self.model.encoder(x, padding_mask=None)
            return x

        x, timings.transformer_ms = self._time_stage(transformer_fn, x)

        # Stage 4: mean pooling
        def pool_fn(x):
            return x.mean(dim=1)

        embeddings, timings.pool_ms = self._time_stage(pool_fn, x)

        self._sync_cuda()
        timings.total_ms = (time.perf_counter() - total_start) * 1000

        return embeddings, timings

    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features without profiling (for compatibility).

        Args:
            waveform: Audio tensor of shape (batch, 1, samples) or (batch, samples)

        Returns:
            Features of shape (batch, embed_dim)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        waveform = waveform.to(self.device)
        features, _ = self.model.extract_features(waveform)
        return features.mean(dim=1)

    def get_frontend_modules(self) -> dict[str, nn.Module]:
        """Get frontend modules for ONNX export.

        Returns dict with:
        - patch_embedding: Conv2d
        - layer_norm: LayerNorm
        - post_extract_proj: Linear (or None)
        """
        return {
            "patch_embedding": self.model.patch_embedding,
            "layer_norm": self.model.layer_norm,
            "post_extract_proj": self.model.post_extract_proj,
        }


def get_memory_stats(device: str = "cuda") -> MemoryStats:
    """Get current memory statistics.

    Args:
        device: Device to get GPU stats for

    Returns:
        MemoryStats with peak GPU and RAM usage
    """
    stats = MemoryStats()

    if device == "cuda" and torch.cuda.is_available():
        stats.peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        stats.allocated_gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    # RAM stats via /proc/self/status (Linux)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    stats.peak_ram_mb = int(line.split()[1]) / 1024
                    break
    except Exception:
        pass

    return stats


def reset_memory_stats(device: str = "cuda"):
    """Reset memory tracking for new benchmark run."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_benchmark(
    model: ProfiledBEATs,
    batch_size: int,
    num_samples: int = 80000,  # 5 seconds at 16kHz
    warmup_iters: int = 5,
    bench_iters: int = 20,
    device: str = "cuda",
) -> BenchmarkResult:
    """Run benchmark with warmup and multiple iterations.

    Args:
        model: ProfiledBEATs model
        batch_size: Number of samples per batch
        num_samples: Audio length in samples (default 5 sec at 16kHz)
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
        device: Device to run on

    Returns:
        BenchmarkResult with averaged timings
    """
    # Create synthetic audio input
    audio = torch.randn(batch_size, num_samples, device=device)

    # Warmup
    for _ in range(warmup_iters):
        _ = model.forward_profiled(audio)

    # Reset memory tracking
    reset_memory_stats(device)

    # Benchmark
    all_timings: list[StageTimings] = []
    for _ in range(bench_iters):
        _, timings = model.forward_profiled(audio)
        all_timings.append(timings)

    # Get memory stats
    memory = get_memory_stats(device)

    # Average timings
    avg_timings = StageTimings(
        fbank_ms=sum(t.fbank_ms for t in all_timings) / len(all_timings),
        patch_embed_ms=sum(t.patch_embed_ms for t in all_timings) / len(all_timings),
        layer_norm_ms=sum(t.layer_norm_ms for t in all_timings) / len(all_timings),
        proj_ms=sum(t.proj_ms for t in all_timings) / len(all_timings),
        transformer_ms=sum(t.transformer_ms for t in all_timings) / len(all_timings),
        pool_ms=sum(t.pool_ms for t in all_timings) / len(all_timings),
        total_ms=sum(t.total_ms for t in all_timings) / len(all_timings),
    )

    # Calculate throughput
    throughput = (batch_size * 1000) / avg_timings.total_ms if avg_timings.total_ms > 0 else 0

    return BenchmarkResult(
        batch_size=batch_size,
        num_samples=num_samples,
        timings=avg_timings,
        memory=memory,
        throughput_samples_per_sec=throughput,
    )


def validate_profiling_overhead(
    model: ProfiledBEATs,
    batch_size: int = 4,
    num_samples: int = 80000,
    iters: int = 20,
    device: str = "cuda",
) -> tuple[float, float, float]:
    """Validate that profiling overhead is acceptable (<5%).

    Args:
        model: ProfiledBEATs model
        batch_size: Batch size to test
        num_samples: Audio length
        iters: Number of iterations
        device: Device to run on

    Returns:
        Tuple of (profiled_ms, unprofiled_ms, overhead_pct)
    """
    audio = torch.randn(batch_size, num_samples, device=device)

    # Warmup
    for _ in range(5):
        _ = model.forward_profiled(audio)
        _ = model.extract_features(audio)

    # Time profiled forward
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = model.forward_profiled(audio)
    if device == "cuda":
        torch.cuda.synchronize()
    profiled_ms = (time.perf_counter() - start) * 1000 / iters

    # Time unprofiled forward
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = model.extract_features(audio)
    if device == "cuda":
        torch.cuda.synchronize()
    unprofiled_ms = (time.perf_counter() - start) * 1000 / iters

    overhead_pct = 100 * (profiled_ms - unprofiled_ms) / unprofiled_ms if unprofiled_ms > 0 else 0

    return profiled_ms, unprofiled_ms, overhead_pct
