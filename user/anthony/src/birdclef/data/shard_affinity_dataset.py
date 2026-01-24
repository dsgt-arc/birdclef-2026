"""Shard Affinity Dataset for efficient multi-worker data loading.

This module provides a Dataset implementation that eliminates the I/O bottleneck
in multi-worker DataLoaders by assigning specific shards to each worker process.

Key Design Principles:
1. Worker Affinity: Each worker owns a disjoint subset of shards
   - Worker 0 gets shards [0, 8, 16, ...], Worker 1 gets [1, 9, 17, ...], etc.
   - No two workers ever access the same shard
   - Eliminates 8x redundant I/O that occurs with naive approaches

2. No Cross-Process Contention: Each worker is fully independent
   - No locks, no shared state, no IPC
   - Each worker maintains its own in-memory shard cache
   - Workers can cache multiple shards (configurable)

3. Aggressive Prefetching: Background thread prefetches next shards
   - Each worker predicts which shard it needs next
   - Prefetch runs in a background thread within the worker process
   - Supports both local NVMe cache and network storage fallback

4. Network-Aware Caching: Optimized for slow source / fast local pattern
   - Source data on network storage (~100 MB/s)
   - Local NVMe cache (~3 GB/s) with limited space
   - On-demand copy with LRU eviction

Usage:
    from shard_affinity_dataset import ShardAffinityDataset, create_dataloader

    dataset = ShardAffinityDataset(
        source_cache_dir="/path/to/network/cache",
        local_cache_dir="/tmp/local_cache",
        shards_per_worker=3,
        prefetch_ahead=2,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=64,
        num_workers=8,
    )
"""

from __future__ import annotations

import json
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset


# Wrapper to make safetensors errors picklable
def safe_load_file(path: str) -> dict[str, torch.Tensor]:
    """Load safetensors file with picklable exceptions."""
    try:
        return load_file(path)
    except Exception as e:
        # Re-raise as standard exception to avoid pickling issues
        raise RuntimeError(f"Failed to load safetensors from {path}: {str(e)}") from None


def safe_open_file(path: str, framework: str = "pt"):
    """Open safetensors file with picklable exceptions."""
    try:
        return safe_open(path, framework=framework)
    except Exception as e:
        raise RuntimeError(f"Failed to open safetensors {path}: {str(e)}") from None


@dataclass
class ShardInfo:
    """Metadata for a single shard file."""

    path: Path
    name: str
    n_samples: int
    index: int  # Global shard index


@dataclass
class WorkerState:
    """Per-worker state initialized in worker_init_fn."""

    worker_id: int
    num_workers: int
    assigned_shards: list[ShardInfo]  # Shards this worker owns
    local_index: list[tuple[int, int]]  # (local_shard_idx, sample_idx)
    shard_cache: OrderedDict[int, dict[str, torch.Tensor]]  # local_shard_idx -> tensors
    max_cached_shards: int
    prefetch_executor: ThreadPoolExecutor | None = None
    prefetch_futures: dict[int, Future] = field(default_factory=dict)
    current_shard_idx: int = -1
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "local_hits": 0,
            "source_loads": 0,
            "prefetch_hits": 0,
            "cache_evictions": 0,
        }
    )


class ShardAffinityDataset(Dataset):
    """Dataset with per-worker shard affinity and aggressive prefetching.

    Each DataLoader worker is assigned a disjoint subset of shards. Workers never
    access shards outside their assignment, eliminating redundant I/O and cache
    contention.

    The dataset operates in two phases:
    1. Main process initialization: Loads shard metadata (fast, header-only)
    2. Worker initialization: Each worker builds its local index and starts prefetching

    Args:
        source_cache_dir: Directory containing safetensors shard files (network storage)
        local_cache_dir: Optional local NVMe cache directory for faster access
        shards_per_worker: Maximum shards to keep in memory per worker (default: 3)
        prefetch_ahead: Number of shards to prefetch ahead (default: 2)
        copy_to_local: Whether to copy shards to local cache before loading
        seed: Random seed for shuffling within shards
    """

    def __init__(
        self,
        source_cache_dir: str | Path,
        local_cache_dir: str | Path | None = None,
        shards_per_worker: int = 3,
        prefetch_ahead: int = 2,
        copy_to_local: bool = True,
        seed: int = 42,
    ):
        self.source_cache_dir = Path(source_cache_dir)
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self.shards_per_worker = shards_per_worker
        self.prefetch_ahead = prefetch_ahead
        self.copy_to_local = copy_to_local
        self.seed = seed

        # Load shard metadata (fast - uses safetensors header API)
        self.shards = self._load_shard_metadata()
        self.total_samples = sum(s.n_samples for s in self.shards)

        # Worker state - initialized in worker_init_fn
        self._worker_state: WorkerState | None = None

        # For single-process mode (num_workers=0)
        self._single_process_index: list[tuple[int, int]] | None = None

    def _load_shard_metadata(self) -> list[ShardInfo]:
        """Load shard metadata using safetensors header API.

        This is fast because it only reads the file headers, not the tensor data.
        Metadata is cached to JSON for instant subsequent runs.
        """
        metadata_path = self.source_cache_dir / "shard_metadata.json"
        shard_files = sorted(self.source_cache_dir.glob("*.safetensors"))

        if not shard_files:
            raise ValueError(f"No safetensors files found in {self.source_cache_dir}")

        # Try loading cached metadata
        cached_metadata = self._load_cached_metadata(metadata_path, shard_files)
        if cached_metadata is not None:
            return cached_metadata

        # Build metadata by reading safetensors headers
        print(f"Building shard metadata for {len(shard_files)} files...")
        shards = []
        for idx, path in enumerate(shard_files):
            try:
                with safe_open_file(str(path), framework="pt") as f:
                    shape = f.get_slice("teacher_embeds").get_shape()
                    n_samples = shape[0]
                shards.append(
                    ShardInfo(path=path, name=path.name, n_samples=n_samples, index=idx)
                )
            except Exception as e:
                print(f"WARNING: Failed to read {path.name}: {e}")
                continue

        # Cache metadata for future runs
        self._save_metadata_cache(metadata_path, shards)

        total = sum(s.n_samples for s in shards)
        print(f"Indexed {len(shards)} shards with {total:,} total samples")
        return shards

    def _load_cached_metadata(
        self, metadata_path: Path, shard_files: list[Path]
    ) -> list[ShardInfo] | None:
        """Try to load metadata from cache."""
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                cached = json.load(f)

            shard_names = {p.name for p in shard_files}
            cached_names = set(cached.keys())

            if not shard_names <= cached_names:
                return None

            shards = []
            for idx, path in enumerate(shard_files):
                if path.name in cached:
                    shards.append(
                        ShardInfo(
                            path=path,
                            name=path.name,
                            n_samples=cached[path.name]["n_samples"],
                            index=idx,
                        )
                    )

            total = sum(s.n_samples for s in shards)
            print(f"Loaded cached metadata: {len(shards)} shards, {total:,} samples")
            return shards
        except Exception as e:
            print(f"Failed to load metadata cache: {e}")
            return None

    def _save_metadata_cache(self, metadata_path: Path, shards: list[ShardInfo]):
        """Save metadata cache for future runs."""
        try:
            data = {s.name: {"n_samples": s.n_samples} for s in shards}
            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved metadata cache to {metadata_path}")
        except Exception as e:
            print(f"WARNING: Could not save metadata cache: {e}")

    def worker_init(self, worker_id: int, num_workers: int):
        """Initialize per-worker state. Called by worker_init_fn.

        This is where the magic happens:
        1. Partition shards among workers using modulo assignment
        2. Build local index containing only this worker's samples
        3. Initialize shard cache and prefetch thread

        Args:
            worker_id: This worker's ID (0 to num_workers-1)
            num_workers: Total number of workers
        """
        # Assign shards to this worker: shard_idx % num_workers == worker_id
        assigned_shards = [s for s in self.shards if s.index % num_workers == worker_id]

        # Build local index: (local_shard_idx, sample_idx_within_shard)
        # Samples are processed sequentially within each shard for cache efficiency
        local_index = []
        for local_idx, shard in enumerate(assigned_shards):
            for sample_idx in range(shard.n_samples):
                local_index.append((local_idx, sample_idx))

        # Shuffle the local index for training diversity
        # Use deterministic seed based on worker_id for reproducibility
        import random

        rng = random.Random(self.seed + worker_id)
        rng.shuffle(local_index)

        # Initialize worker state
        self._worker_state = WorkerState(
            worker_id=worker_id,
            num_workers=num_workers,
            assigned_shards=assigned_shards,
            local_index=local_index,
            shard_cache=OrderedDict(),
            max_cached_shards=self.shards_per_worker,
            prefetch_executor=ThreadPoolExecutor(max_workers=1),
            prefetch_futures={},
        )

        # Ensure local cache directory exists
        if self.local_cache_dir and self.copy_to_local:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Start prefetching first few shards
        for i in range(min(self.prefetch_ahead, len(assigned_shards))):
            self._schedule_prefetch(i)

    def _get_shard_path(self, shard: ShardInfo) -> Path:
        """Get the best available path for a shard (local preferred)."""
        if self.local_cache_dir and self.copy_to_local:
            local_path = self.local_cache_dir / shard.name
            if local_path.exists():
                return local_path

        return shard.path

    def _copy_shard_to_local(self, shard: ShardInfo) -> Path:
        """Copy a shard from source to local cache."""
        if not self.local_cache_dir or not self.copy_to_local:
            return shard.path

        local_path = self.local_cache_dir / shard.name
        if local_path.exists():
            return local_path

        try:
            shutil.copy2(shard.path, local_path)
            return local_path
        except Exception as e:
            print(f"WARNING: Failed to copy {shard.name} to local: {e}")
            return shard.path

    def _load_shard_tensors(self, local_shard_idx: int) -> dict[str, torch.Tensor]:
        """Load tensors from a shard file.

        Checks prefetch cache first, then loads from disk.
        Manages LRU eviction of cached shards.
        """
        state = self._worker_state
        assert state is not None, "worker_init must be called first"

        # Check if already cached
        if local_shard_idx in state.shard_cache:
            state.shard_cache.move_to_end(local_shard_idx)
            return state.shard_cache[local_shard_idx]

        # Check if prefetch completed
        if local_shard_idx in state.prefetch_futures:
            future = state.prefetch_futures.pop(local_shard_idx)
            try:
                tensors = future.result(timeout=30)
                if tensors is not None:
                    state.stats["prefetch_hits"] += 1
                    self._cache_shard(local_shard_idx, tensors)
                    return tensors
            except Exception as e:
                print(f"WARNING: Prefetch failed for shard {local_shard_idx}: {e}")

        # Load from disk
        shard = state.assigned_shards[local_shard_idx]
        path = self._get_shard_path(shard)

        # Copy to local if needed and configured
        if self.copy_to_local and self.local_cache_dir:
            path = self._copy_shard_to_local(shard)

        try:
            tensors = safe_load_file(str(path))
            if path.parent == self.local_cache_dir:
                state.stats["local_hits"] += 1
            else:
                state.stats["source_loads"] += 1
            self._cache_shard(local_shard_idx, tensors)
            return tensors
        except Exception as e:
            print(f"ERROR: Failed to load shard {shard.name}: {e}")
            raise

    def _cache_shard(self, local_shard_idx: int, tensors: dict[str, torch.Tensor]):
        """Add shard to cache, evicting LRU if necessary."""
        state = self._worker_state
        assert state is not None

        # Evict oldest if at capacity
        while len(state.shard_cache) >= state.max_cached_shards:
            oldest_idx = next(iter(state.shard_cache))
            del state.shard_cache[oldest_idx]
            state.stats["cache_evictions"] += 1

        state.shard_cache[local_shard_idx] = tensors

    def _prefetch_shard(self, local_shard_idx: int) -> dict[str, torch.Tensor] | None:
        """Worker function for background prefetching."""
        state = self._worker_state
        if state is None or local_shard_idx >= len(state.assigned_shards):
            return None

        shard = state.assigned_shards[local_shard_idx]

        # Copy to local first if configured
        if self.copy_to_local and self.local_cache_dir:
            path = self._copy_shard_to_local(shard)
        else:
            path = self._get_shard_path(shard)

        try:
            return safe_load_file(str(path))
        except Exception as e:
            print(f"WARNING: Prefetch load failed for {shard.name}: {e}")
            return None

    def _schedule_prefetch(self, local_shard_idx: int):
        """Schedule background prefetch for a shard."""
        state = self._worker_state
        if state is None or state.prefetch_executor is None:
            return

        # Skip if already cached or already being prefetched
        if local_shard_idx in state.shard_cache:
            return
        if local_shard_idx in state.prefetch_futures:
            return
        if local_shard_idx >= len(state.assigned_shards):
            return

        future = state.prefetch_executor.submit(self._prefetch_shard, local_shard_idx)
        state.prefetch_futures[local_shard_idx] = future

    def __len__(self) -> int:
        """Return total number of samples across all shards."""
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        In multi-worker mode, idx is mapped to this worker's local index.
        The DataLoader's sampler distributes indices across workers, but we
        remap them to our local shard assignment.
        """
        state = self._worker_state

        # Single-process mode (num_workers=0)
        if state is None:
            return self._getitem_single_process(idx)

        # Map global idx to local index
        local_idx = idx % len(state.local_index)
        local_shard_idx, sample_idx = state.local_index[local_idx]

        # Trigger prefetch when entering a new shard region
        if local_shard_idx != state.current_shard_idx:
            state.current_shard_idx = local_shard_idx
            # Prefetch next few shards
            for offset in range(1, self.prefetch_ahead + 1):
                self._schedule_prefetch(local_shard_idx + offset)

        # Load shard and extract sample
        tensors = self._load_shard_tensors(local_shard_idx)

        static_embeds = tensors["static_embeds"][sample_idx].float()
        teacher_embeds = tensors["teacher_embeds"][sample_idx].float()

        return static_embeds, teacher_embeds

    def _getitem_single_process(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback for single-process mode (num_workers=0)."""
        # Build index lazily
        if self._single_process_index is None:
            self._single_process_index = []
            for shard_idx, shard in enumerate(self.shards):
                for sample_idx in range(shard.n_samples):
                    self._single_process_index.append((shard_idx, sample_idx))

        shard_idx, sample_idx = self._single_process_index[idx]
        shard = self.shards[shard_idx]
        path = self._get_shard_path(shard)
        tensors = safe_load_file(str(path))

        static_embeds = tensors["static_embeds"][sample_idx].float()
        teacher_embeds = tensors["teacher_embeds"][sample_idx].float()

        return static_embeds, teacher_embeds

    def get_worker_stats(self) -> dict[str, Any]:
        """Get statistics for the current worker."""
        if self._worker_state is None:
            return {"mode": "single_process"}

        state = self._worker_state
        return {
            "worker_id": state.worker_id,
            "num_workers": state.num_workers,
            "assigned_shards": len(state.assigned_shards),
            "local_samples": len(state.local_index),
            "cached_shards": len(state.shard_cache),
            "pending_prefetches": len(state.prefetch_futures),
            **state.stats,
        }

    def reset_epoch(self, epoch: int):
        """Reshuffle local index for a new epoch.

        Call this at the start of each training epoch to get different
        sample ordering while maintaining shard affinity.
        """
        import random

        if self._worker_state is not None:
            state = self._worker_state
            rng = random.Random(self.seed + epoch * 1000 + state.worker_id)
            rng.shuffle(state.local_index)
            state.current_shard_idx = -1

    def cleanup(self):
        """Clean up resources (call at end of training)."""
        if self._worker_state is not None and self._worker_state.prefetch_executor:
            self._worker_state.prefetch_executor.shutdown(wait=False)


def worker_init_fn(worker_id: int):
    """PyTorch DataLoader worker_init_fn.

    This function is called once in each worker process at startup.
    It initializes the per-worker shard assignment and prefetching.

    Usage:
        DataLoader(dataset, worker_init_fn=worker_init_fn, ...)
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return  # Single-process mode

    dataset = info.dataset
    if hasattr(dataset, "worker_init"):
        dataset.worker_init(worker_id, info.num_workers)


def create_dataloader(
    dataset: ShardAffinityDataset,
    batch_size: int = 64,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with proper worker affinity settings.

    This helper function configures the DataLoader with optimal settings
    for the ShardAffinityDataset.

    Args:
        dataset: ShardAffinityDataset instance
        batch_size: Batch size per iteration
        num_workers: Number of worker processes (each gets disjoint shards)
        shuffle: Whether to shuffle (handled by dataset, but enables sampler)
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        drop_last: Drop incomplete final batch

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )


def collate_fn_skip_none(
    batch: list[tuple[torch.Tensor, torch.Tensor] | None],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Collate function that filters out None samples.

    Use this if your dataset might return None for corrupted/missing samples.

    Args:
        batch: List of (static_embeds, teacher_embeds) tuples or None

    Returns:
        Stacked tensors or None if all samples were invalid
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    static_embeds = torch.stack([item[0] for item in batch])
    teacher_embeds = torch.stack([item[1] for item in batch])
    return static_embeds, teacher_embeds


# Convenience function for pre-warming local cache
def prewarm_local_cache(
    source_cache_dir: str | Path,
    local_cache_dir: str | Path,
    num_workers: int = 8,
    max_shards: int | None = None,
) -> int:
    """Copy shards to local NVMe cache before training starts.

    This can be run as a separate process before training to ensure
    the local cache is warm when training begins.

    Args:
        source_cache_dir: Source directory containing safetensors files
        local_cache_dir: Local cache directory (typically on NVMe)
        num_workers: Number of parallel copy workers
        max_shards: Maximum shards to copy (None for all)

    Returns:
        Number of shards copied
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    source = Path(source_cache_dir)
    local = Path(local_cache_dir)
    local.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(source.glob("*.safetensors"))
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    # Filter out already-copied shards
    to_copy = [f for f in shard_files if not (local / f.name).exists()]

    if not to_copy:
        print(f"All {len(shard_files)} shards already in local cache")
        return 0

    print(f"Pre-warming local cache: {len(to_copy)} shards to copy...")

    def copy_shard(src: Path) -> bool:
        try:
            shutil.copy2(src, local / src.name)
            return True
        except Exception as e:
            print(f"WARNING: Failed to copy {src.name}: {e}")
            return False

    copied = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(copy_shard, f): f for f in to_copy}
        for future in as_completed(futures):
            if future.result():
                copied += 1

    print(f"Pre-warmed {copied}/{len(to_copy)} shards to {local}")
    return copied
