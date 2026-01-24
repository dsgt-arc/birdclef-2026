# Performance Monitoring Design for Distillation Training

**Status**: active
**Created**: 2026-01-23
**Related**: proposal.md (Experiment 003)

## Objective

Design a comprehensive monitoring strategy to identify and resolve performance bottlenecks in the distillation training pipeline, with focus on saturating GPU utilization by optimizing the data loading path (network → scratch disk → local SSD → GPU).

## Background

The distillation pipeline processes 687GB of cached teacher embeddings from scratch storage (`~/scratch/birdclef-2026`) with optional copying to local NVMe (`$TMPDIR`). With 8 DataLoader workers, aggressive prefetching, and shard affinity, we need to determine:

1. **Is the GPU starved?** (< 80% utilization indicates dataloader bottleneck)
2. **Where is the bottleneck?** (Network I/O, Disk I/O, Deserialization, or GPU compute)
3. **Are we optimally using local SSD?** (Is copying to `$TMPDIR` worth it?)

## Monitoring Strategy

### Phase 1: Unified TensorBoard Dashboard (PyTorch Lightning)

Instead of disparate CLI tools and log files, we will leverage **PyTorch Lightning** to consolidate all training and system metrics into a single TensorBoard timeline. This allows direct correlation between system spikes (e.g., IO wait) and training steps.

#### 1. System Metrics (`DeviceStatsMonitor`)
Use the built-in callback to log CPU/GPU usage automatically.

```python
from lightning.pytorch.callbacks import DeviceStatsMonitor

trainer = Trainer(
    callbacks=[DeviceStatsMonitor(cpu_stats=True)],
    logger=TensorBoardLogger("lightning_logs", name="distill_500k"),
    ...
)
```

**Metrics Captured**:
- `gpu_id:0/utilization` (GPU Load)
- `gpu_id:0/memory.used` (VRAM)
- `on_epoch_start` / `on_step_end` timings

#### 2. Custom Data Loading Metrics (`Callback`)
Since standard callbacks don't track disk I/O, we will add a custom callback to log `psutil` IO counters and `ShardAffinityDataset` stats.

```python
class DataPerfMonitor(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log specific dataset stats if available (e.g. from our custom dataset)
        if hasattr(trainer.datamodule.train_dataset, 'get_worker_stats'):
            stats = trainer.datamodule.train_dataset.get_worker_stats()
            pl_module.log_dict({f"data/{k}": v for k, v in stats.items()})
```

### Phase 2: Deep Profiling (PyTorch Profiler + py-spy)

When the high-level dashboard indicates a bottleneck, we use specialized tools to drill down.

#### 2a. Integrated PyTorch Profiler
We will use Lightning's `PyTorchProfiler` to capture traces that open directly in the TensorBoard "Profile" plugin. This shows the exact breakdown of Kernel vs. DataLoader vs. CPU time.

```python
from lightning.pytorch.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="profile_trace",
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler,
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)

trainer = Trainer(profiler=profiler, ...)
```

#### 2b. Ad-hoc py-spy Flamegraphs
For non-PyTorch CPU bottlenecks (e.g., `safetensors` deserialization, `shutil.copy` stalls), `py-spy` is the tool of choice.

```bash
# Snapshot
py-spy record -o flamegraph.svg --pid <PID> --duration 60

# Live View
py-spy top --pid <PID>
```

### Phase 3: A/B Testing Data Paths

Refined experiment matrix to isolate variables:

| Test | Config | Hypothesis |
|------|--------|------------|
| **Baseline** | `copy=True`, `workers=8`, `prefetch=4` | Reference performance |
| **Direct Read** | `copy=False` | Is untarring/copying slower than network reads? |
| **High Prefetch** | `copy=True`, `prefetch=8` | Can we hide the copy latency? |
| **Affinity** | `workers=4` + `taskset` | Variable: CPU contention |

## Expected Artifacts

All data will now be centralized in the `lightning_logs/` directory:

```
lightning_logs/
└── version_0/
    ├── events.out.tfevents.*  # Contains Loss, GPU Util, CPU Util, Custom Metrics
    └── profiler_logs/         # PyTorch Profiler traces (viewable in TB)
```

## Analysis & Interpretation

### Decision Tree

```
GPU Util < 80%?
├─ YES: DataLoader is bottleneck
│   ├─ Disk I/O wait high?
│   │   ├─ YES: Disk read is slow
│   │   │   ├─ copy_to_local=False → Try direct network read
│   │   │   └─ Increase prefetch_factor
│   │   └─ NO: Deserialization is slow
│   │       ├─ Use memmapped tensors
│   │       └─ Pre-load shards at epoch start
│   └─ Network I/O high?
│       ├─ YES: Network bandwidth saturated
│       │   └─ Increase local caching
│       └─ NO: CPU deserialization bottleneck
│           └─ Reduce num_workers, increase prefetch_factor
└─ NO: GPU is saturated (good!)
    └─ Check if we can increase batch size
```

### Target Performance

Based on hardware specs (RTX 6000, 8-core CPU, NVMe SSD):

| Component | Target | Indicates Health |
|-----------|--------|------------------|
| GPU Utilization | > 85% | GPU not starved |
| DataLoader Iter Time | < 50ms/batch | Fast enough for training |
| Disk Read Throughput | > 500 MB/s (NVMe) | Not I/O bound |
| Network Throughput | < 5 Gbps | Not network limited |
| Samples/sec | > 500 (batch=64) | Overall pipeline efficient |

## Implementation Plan

1. **Immediate**: Add PyTorch profiler to next training run
2. **Short-term**: Instrument ShardAffinityDataset with timing
3. **Medium-term**: Run A/B tests on data loading configs
4. **Long-term**: Build dashboard for real-time monitoring

## Expected Outcomes

After monitoring, we should be able to answer:

1. **Primary bottleneck**: Network, Disk, CPU decompression, or GPU
2. **Optimal config**: Which combination of `copy_to_local`, `num_workers`, `prefetch_factor` maximizes throughput
3. **Hardware limits**: Whether we're hitting theoretical limits (PCIe bandwidth, network bandwidth)

## Next Steps

- [ ] Fix current pickling error with safetensors workers
- [ ] Add PyTorch profiler to training script
- [ ] Run baseline monitoring with current config
- [ ] Analyze profiler output
- [ ] Run A/B tests with different configs
- [ ] Document findings in results.md
