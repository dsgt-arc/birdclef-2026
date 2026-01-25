# Shard Affinity Dataset: Efficient Multi-Worker Data Loading

## Overview

`ShardAffinityDataset` is a PyTorch Dataset implementation that eliminates I/O bottlenecks in multi-worker DataLoaders through worker-shard affinity and two-level prefetching. It replaces the earlier `StreamingCacheDataset` with a more scalable architecture.

Key insight: By assigning each worker a **disjoint** subset of shards, we eliminate cross-worker contention and redundant I/O entirely.

## Architecture

### Module Structure

```
birdclef/
├── caching.py                    # TeacherCacher - generates cache from audio
├── data/
│   └── shard_affinity_dataset.py # ShardAffinityDataset - loads cache for training
└── lightning/
    └── distillation_datamodule.py # PyTorch Lightning integration
```

### Cache Format (v2)

Each shard is a `.safetensors` file containing:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `teacher_embeds` | (N, 248, 768) | fp16 | Contextualized embeddings from BEATs |
| `token_ids` | (N, 248) | int16 | Discrete token IDs for embedding lookup |

**Static embeddings are NOT stored** - they're reconstructed at load time from `token_ids` using a lookup table (~24MB). This reduces cache size by 50%.

```python
# At load time:
token_ids = tensors["token_ids"][sample_idx].long()  # (248,)
static_embeds = static_embeddings_lut[token_ids]     # (248, 768)
```

### Worker Affinity Design

```
8 workers, 143 shards:

Worker 0: shards [0, 8, 16, 24, 32, ...]  (18 shards)
Worker 1: shards [1, 9, 17, 25, 33, ...]  (18 shards)
Worker 2: shards [2, 10, 18, 26, 34, ...] (18 shards)
...
Worker 7: shards [7, 15, 23, 31, 39, ...] (18 shards)
```

Benefits:
- **No contention**: Workers never access the same shard
- **No redundant I/O**: Each shard loaded exactly once per epoch
- **Predictable access**: Enables effective prefetching

### Two-Level Prefetching

```
                    ┌─────────────────────────────────────────────┐
                    │          Worker Process                      │
                    ├─────────────────────────────────────────────┤
                    │                                              │
                    │  ┌────────────────┐   ┌──────────────────┐  │
                    │  │ copy_executor  │   │ prefetch_executor│  │
                    │  │ (1 thread)     │   │ (2 threads)      │  │
                    │  └───────┬────────┘   └────────┬─────────┘  │
                    │          │                     │             │
                    │          ▼                     ▼             │
                    │  ┌────────────────┐   ┌──────────────────┐  │
Network Storage ───►│  │  Local NVMe    │──►│  Memory Cache    │  │
(~100 MB/s)         │  │  (~3 GB/s)     │   │  (3 shards)      │  │
                    │  └────────────────┘   └────────┬─────────┘  │
                    │                                │             │
                    │                                ▼             │
                    │                        Return to GPU         │
                    └─────────────────────────────────────────────┘
```

1. **copy_executor**: Background thread copies upcoming shards from network to local NVMe
2. **prefetch_executor**: Background threads load shards from disk into memory

### Data Flow

```
__getitem__(idx)
    │
    ├─ Map idx to (local_shard_idx, sample_idx) via worker's local_index
    │
    ├─ New shard? Schedule prefetch for next N shards
    │
    ├─ 50% through shard? Early prefetch trigger for shard N+1
    │
    ├─ Load shard:
    │   ├─ Check memory cache → hit: return immediately
    │   ├─ Check prefetch futures → hit: wait & cache result
    │   └─ Load from disk (local NVMe preferred, network fallback)
    │
    ├─ Reconstruct static_embeds from token_ids via LUT
    │
    └─ Return (static_embeds, teacher_embeds)
```

## Usage

### Generating Cache (TeacherCacher)

```python
from birdclef.caching import TeacherCacher

cacher = TeacherCacher(
    encoder_path="path/to/BEATs_iter3.pt",
    tokenizer_path="path/to/Tokenizers_iter3.pt",
)

# Process single shard
tensors = cacher.process_shard(tar_path)

# Process multiple shards with progress
stats = cacher.cache_shards(tar_files, output_dir)
```

### Loading Cache (ShardAffinityDataset)

```python
from birdclef.data.shard_affinity_dataset import (
    ShardAffinityDataset,
    create_dataloader,
)

dataset = ShardAffinityDataset(
    source_cache_dir="/path/to/network/cache",
    static_embeddings_path="/path/to/static_embeddings.npy",
    local_cache_dir="/tmp/local_cache",
    shards_per_worker=3,
    prefetch_ahead=2,
)

dataloader = create_dataloader(
    dataset,
    batch_size=64,
    num_workers=8,
)
```

### PyTorch Lightning Integration

```python
from birdclef.lightning.distillation_datamodule import DistillationDataModule

datamodule = DistillationDataModule(
    source_cache_dir="/path/to/cache",
    static_embeddings_path="/path/to/static_embeddings.npy",
    local_cache_dir="/tmp/cache",
    batch_size=64,
    num_workers=8,
    shards_per_worker=3,
)
```

## Configuration

### Default Configuration

```python
ShardAffinityDataset(
    source_cache_dir=...,
    static_embeddings_path=...,
    local_cache_dir=None,       # No local caching
    shards_per_worker=3,        # Keep 3 shards in memory per worker
    prefetch_ahead=2,           # Prefetch 2 shards ahead
    copy_to_local=True,         # Copy to local NVMe if local_cache_dir set
    seed=42,                    # Deterministic shuffling
)
```

### Configuration Tuning

| Scenario | shards_per_worker | prefetch_ahead | local_cache_dir |
|----------|-------------------|----------------|-----------------|
| Fast NVMe, plenty RAM | 5 | 3 | Set to TMPDIR |
| Limited RAM | 2 | 2 | Set to TMPDIR |
| No local storage | 3 | 2 | None |
| Very slow network | 4 | 4 | Set to TMPDIR |

### Memory Requirements

Per-worker memory for shard cache:

```
shards_per_worker × samples_per_shard × (teacher_embeds + token_ids)
= 3 × 3500 × (248×768×2 + 248×2) bytes
≈ 3 × 3500 × 381KB
≈ 4 GB per worker
```

With 8 workers: ~32 GB total for shard cache (shared via memory mapping).

## Performance Characteristics

### I/O Efficiency

| Approach | Redundant I/O | Contention |
|----------|---------------|------------|
| Naive (no affinity) | 8x (each worker loads all shards) | High |
| StreamingCacheDataset (old) | 1x but with LRU thrashing | Medium |
| **ShardAffinityDataset** | **1x, no thrashing** | **None** |

### Access Latency

| Source | Latency |
|--------|---------|
| Memory cache (current shard) | ~1ms |
| Local NVMe (warm shard) | ~20-50ms |
| Network storage (cold shard) | ~100-300ms |

### Cache Behavior by Epoch

```
Epoch 1: Warming phase
  - Workers copy assigned shards to local NVMe in background
  - First access to each shard loads from network
  - Subsequent accesses use local NVMe

Epoch 2+: Warm phase
  - All shards already on local NVMe
  - 95%+ local hits
  - GPU never starved for data
```

## Monitoring

### Worker Statistics

```python
stats = dataset.get_worker_stats()
# Returns:
{
    "worker_id": 0,
    "assigned_shards": 18,
    "local_samples": 63000,
    "cached_shards": 3,
    "pending_prefetches": 2,
    "local_hits": 4567,
    "source_loads": 123,
    "prefetch_hits": 890,
    "cache_evictions": 15,
    "background_copies": 18,
}
```

### Key Metrics

**Good signs:**
- `local_hits >> source_loads` after epoch 1
- `prefetch_hits` increasing (prefetch working)
- `cache_evictions` low and stable

**Bad signs:**
- High `source_loads` in later epochs (local copy failing)
- Zero `prefetch_hits` (prefetch not triggering)
- `cache_evictions` equals `access_count` (cache thrashing)

## Storage Requirements

For AudioSet-500k with 143 shards:

| Component | Size |
|-----------|------|
| Cache (v2 format) | ~350 GB |
| Static embeddings LUT | ~24 MB |
| Per-shard | ~2.5 GB |
| Local NVMe (recommended) | 100+ GB |

## Thread Safety

All shared state protected by locks:

- `_copy_locks`: Per-shard locks for atomic file copies
- `threading.Lock` within each worker for shard cache

Workers are fully independent - no cross-process coordination required.

### Atomic File Operations

```python
# Write to temp file, then atomic rename
fd, tmp_path = tempfile.mkstemp(dir=local_cache_dir)
shutil.copy2(source, tmp_path)
tmp_path.rename(final_path)  # Atomic on POSIX
```

This prevents reading incomplete files during concurrent access.

## Epoch Reshuffling

```python
# Called at epoch start
dataset.reset_epoch(epoch)
```

Reshuffles the local index within each worker using deterministic seed:
```python
seed = base_seed + epoch * 1000 + worker_id
```

Maintains shard affinity while varying sample order.

## Files

| File | Purpose |
|------|---------|
| `birdclef/caching.py` | TeacherCacher for generating v2 cache |
| `birdclef/data/shard_affinity_dataset.py` | ShardAffinityDataset and helpers |
| `birdclef/lightning/distillation_datamodule.py` | Lightning DataModule wrapper |

## Comparison with Old Implementation

| Feature | StreamingCacheDataset (old) | ShardAffinityDataset (new) |
|---------|----------------------------|---------------------------|
| Worker assignment | Any worker, any shard | Disjoint shard assignment |
| Cache strategy | Global LRU with eviction | Per-worker memory cache |
| Prefetch | Single background thread | Two-level (copy + load) |
| Cache format | v1 (with static_embeds) | v2 (reconstructed at runtime) |
| Contention | Lock-based coordination | Lock-free (disjoint shards) |
| I/O efficiency | LRU thrashing possible | Optimal (1x per shard) |

## References

- [WebDataset](https://github.com/webdataset/webdataset) - Shard-based dataset format
- [safetensors](https://github.com/huggingface/safetensors) - Fast tensor serialization
- PyTorch DataLoader worker architecture
