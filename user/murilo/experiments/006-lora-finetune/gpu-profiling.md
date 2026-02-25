# GPU Profiling Notes — Experiment 006

Performance analysis from GPU utilization checks on LoRA fine-tuning runs.
For general PACE GPU best practices, see `docs/references/pace-gpu-workflows.md`.

## GPU Check Results

Two 1-epoch GPU checks using `--gpu-check 1` on the `eat_lora_attn` config:

| Metric | PEFT (job 4233484) | Adapters (interactive 4282919) |
|--------|-------------------|-------------------------------|
| Data Time (avg) | **68.1 ms** | 3.4 ms |
| Compute Time (avg) | 116.7 ms | **176.1 ms** |
| Data Loading % | **35.5%** | 1.9% |
| Throughput | 182 samples/s | 178 samples/s |
| Memory Allocated | 0.61 GB (2.6%) | 0.61 GB (2.4%) |
| Memory Reserved | 5.36 GB | 5.12 GB |
| GPU Utilization | 90.6% | 88.3% |

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
See `docs/references/pace-gpu-workflows.md` for details on PACE storage tiers.

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

| batch_size | accum | steps/epoch | est. reserved | fits 24GB? |
|-----------|-------|------------|---------------|------------|
| 32 | 8 | 5,411 | ~5.4 GB | yes (current) |
| 64 | 4 | 2,706 | ~10.3 GB | likely yes |
| 128 | 2 | 1,353 | ~20.2 GB | tight |
| 256 | 1 | 676 | ~40 GB | OOM |

**batch_size=64 with accum=4** is the safe next step — halves per-epoch overhead with
the same effective batch size.

## Potential Optimizations (Ranked by Impact)

### 1. Increase batch_size to 64, reduce accum to 4 (est. ~40% faster)

The main bottleneck is 5,411 forward+backward passes per epoch. Doubling batch_size
halves that to 2,706 — fewer DataLoader calls, fewer Python iterations, better GPU
parallelism. Same effective batch size (256).

```yaml
data:
  batch_size: 64          # was 32
training:
  accumulate_grad_batches: 4  # was 8
```

Needs a `--gpu-check 1` to confirm no OOM (~10 GB estimated reserved).

### 2. Precompute fbank features (eliminates CPU transform overhead)

The `waveform_to_fbank` transform runs on CPU workers every `__getitem__` call. Saving
precomputed fbank tensors would eliminate both disk I/O and CPU transform cost. Bigger
engineering effort but the most impactful for repeated experiments.

### 3. Copy audio data to `$TMPDIR` (helps on cold nodes only)

The BirdCLEF 2025 dataset is ~7GB — small enough that the OS page cache absorbs it
after the first epoch. On a warm node, NFS data loading is already ~3ms/step (1.9% of
step time), so the copy would have negligible impact.

However, on a **cold SLURM node** (first job on that machine, or evicted page cache),
data loading jumps to ~68ms/step (35% of step time). Copying to `$TMPDIR` guarantees
consistently fast reads regardless of cache state:

```bash
LOCAL_DATA="$TMPDIR/data/birdclef-2025"
mkdir -p "$TMPDIR/data"
cp -r ~/ps-clef2026_birdclef-0/data/birdclef-2025 "$LOCAL_DATA"
```

Worth doing for robustness, but not the primary bottleneck for this experiment.

## Outputs

- `outputs/gpu_check/eat-lora-attn_4233484.{txt,png}` — PEFT (interactive session, cold page cache)
- `outputs/gpu_check/adapters-eat-lora-attn_4282919.{txt,png}` — Adapters (interactive session, warm page cache)
