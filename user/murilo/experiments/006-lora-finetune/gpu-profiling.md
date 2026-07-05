# GPU Profiling Notes — Experiment 006

Performance analysis from GPU utilization checks on LoRA fine-tuning runs.
For current PACE/SLURM operational guidance, see `docs/operations.md`.

## GPU Check Results

Three 1-epoch GPU checks using `--gpu-check 1` on `eat_lora_attn` configs:

| Metric | PEFT bs=32 (4233484) | Adapters bs=32 (4282919) | PEFT bs=64 (4282919) |
|--------|---------------------|-------------------------|---------------------|
| Steps/epoch | 5,411 | 5,411 | 2,705 |
| Data Time (avg) | **68.1 ms** | 3.4 ms | **110.7 ms** |
| Compute Time (avg) | 116.7 ms | **176.1 ms** | 203.2 ms |
| Data Loading % | **35.5%** | 1.9% | **31.8%** |
| Throughput | 182 samples/s | 178 samples/s | **232 samples/s** |
| Memory Allocated | 0.61 GB (2.6%) | 0.61 GB (2.4%) | 0.62 GB (2.5%) |
| Memory Reserved | 5.36 GB | 5.12 GB | **9.84 GB** |
| GPU Utilization | 90.6% | 88.3% | **92.5%** |

### Why the data loading difference?

Both checks ran on interactive SLURM sessions (`sacct` confirms `interactive` job type).
The PEFT check ran on an earlier session (job 4233484, node atl1-1-03-007) and the
adapters check on a later session (job 4282919, node atl1-1-02-004). The 20x data
loading gap (68ms vs 3ms) is likely due to **OS page cache warmth** — the adapters
session had more audio data cached in RAM from prior work on that node, while the PEFT
session may have been run before the data was fully cached, or on a node with more I/O
contention.

The takeaway: data loading from NFS varies widely (3-68ms) depending on page cache state.
Copying data to `$TMPDIR` (local NVMe) eliminates this variability entirely.
See `docs/operations.md` for the current `$TMPDIR` and cache guidance.

### Why is adapters compute slower?

Adapters uses `LoRAMergedLinear` with per-matrix (Q/K/V) enable flags and grouped
convolutions for the fused QKV layer. PEFT applies a single LoRA adapter to the entire
fused `qkv` linear. The finer-grained handling adds ~50% compute overhead per step
(176ms vs 117ms), though throughput is similar because adapters has less data loading
overhead on warm cache.

## Training Time Breakdown

With batch_size=32 and 5,411 steps/epoch (173K training samples):

```
Per epoch:  5,411 steps × ~185ms/step     ≈ 17 min (training)
            671 val steps × ~100ms/step    ≈  1 min (validation)
                                           ≈ 18 min total

Full run:   30 epochs × 18 min            ≈ 9 hours
            + test evaluation              ≈ 10 min
                                           ≈ 9-10 hours
```

This matches the observed 9-11 hour training times across all runs.

## Where Time Is Spent

Ranges reflect variability from NFS page cache state (cold vs warm):

| Component | Time/epoch (cold) | Time/epoch (warm) | Optimization |
|-----------|------------------|-------------------|--------------|
| Data loading | ~6 min (37%) | ~0.3 min (2%) | Copy data to `$TMPDIR` |
| Forward + backward | ~10.5 min (58%) | ~16 min (89%) | Increase batch_size |
| Validation | ~1 min (6%) | ~1 min (6%) | — |
| **Total** | **~17.5 min** | **~17.3 min** | |

Note: total per-epoch time is similar because the adapters backend is ~50% slower on
compute, offsetting the data loading gains. For PEFT with warm cache, epochs would be
~11 min (117ms compute + 3ms data = 120ms/step × 5411 = ~10.8 min + val).

## Batch Size vs Gradient Accumulation

Current config: `batch_size=32, accumulate_grad_batches=8` → effective batch = 256.

The original proposal tried `batch_size=256` without accumulation — **OOM'd**. LoRA still
backprops through all 90M encoder params, so activation memory ≈ full fine-tuning.

Memory headroom analysis (`memory_reserved` ≈ actual peak usage):

| batch_size | accum | steps/epoch | reserved (measured) | fits 24GB? |
|-----------|-------|------------|---------------------|------------|
| 32 | 8 | 5,411 | 5.4 GB | yes (current) |
| 64 | 4 | 2,706 | **9.8 GB** (tested) | yes |
| 128 | 2 | 1,353 | ~20 GB (est.) | tight |
| 256 | 1 | 676 | — | OOM (confirmed) |

## Potential Optimizations

### batch_size=64 — tested, not worth it

GPU check with `batch_size=64, accum=4` confirmed it fits in memory (9.8 GB reserved)
but **per-epoch time only improves ~18%** (from ~17 min to ~14 min). Half the steps but
each step takes ~70% longer — the forward+backward through the 90M EAT encoder scales
linearly with batch size, so doubling the batch roughly doubles per-step compute.

The real bottleneck is the encoder itself, not Python/DataLoader overhead.

### Precompute fbank features (best remaining option)

The `waveform_to_fbank` transform runs on CPU workers every `__getitem__` call. Saving
precomputed fbank tensors would eliminate both disk I/O and CPU transform cost. Bigger
engineering effort but the most impactful for repeated experiments on this dataset.

### Copy audio data to `$TMPDIR` (marginal for this dataset)

The BirdCLEF 2025 dataset is ~7GB — small enough that the OS page cache absorbs it
after the first epoch. On a warm node, NFS data loading is already ~3ms/step (1.9% of
step time). On a cold node it jumps to ~68-111ms/step. Copying to `$TMPDIR` guarantees
consistent reads but doesn't address the core compute bottleneck.

### Conclusion

Training time for LoRA fine-tuning of EAT-base on BirdCLEF 2025 is fundamentally
limited by the encoder forward+backward pass (~117-203ms per step depending on backend
and batch size). With ~173K training samples and early stopping at ~30 epochs, 9-11
hours on a single RTX 6000 is roughly the floor for this architecture. Meaningful
speedups would require model-level changes (smaller encoder, gradient checkpointing,
multi-GPU) rather than data pipeline tuning.

## Outputs

- `outputs/gpu_check/eat-lora-attn_4233484.{txt,png}` — PEFT bs=32 (interactive, cold page cache)
- `outputs/gpu_check/adapters-eat-lora-attn_4282919.{txt,png}` — Adapters bs=32 (interactive, warm page cache)
- `outputs/gpu_check/eat-lora-attn-bs64_4282919.{txt,png}` — PEFT bs=64 (interactive, cold page cache)
