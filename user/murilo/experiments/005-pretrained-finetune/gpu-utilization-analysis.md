# GPU Utilization Analysis — Experiment 005

## How GPU Training Works (quick primer)

A GPU training loop is a **producer-consumer pipeline**:

```
CPU (producer)                          GPU (consumer)
─────────────                          ──────────────
Load audio from disk ──┐
Resample / transform   │  ──batch──>   Forward pass (compute predictions)
Collate into batch   ──┘               Backward pass (compute gradients)
                                       Optimizer step (update weights)
```

The GPU can only work when a batch is ready. If the CPU can't produce batches
fast enough, the GPU sits **idle** waiting — this is called **data starvation**.

Key concepts:
- **GPU Utilization %** — how much of the time the GPU's compute cores are active
  (100% = fully busy, 0% = idle waiting for data)
- **VRAM (GPU memory)** — stores the model weights, activations, and gradients.
  Low usage means you could increase batch size or use a larger model.
- **Data time** — time between batches (how long the GPU waits for the next batch)
- **Compute time** — time for forward + backward + optimizer (actual GPU work)
- **num_workers** — parallel CPU processes loading data. More workers = faster
  batch production, but capped by available CPU cores.
- **prefetch** — PyTorch DataLoader pre-loads the next batch(es) in the background.
  Hides data loading latency when the pipeline is balanced, but can't help when
  data production is consistently slower than GPU compute.

**The goal:** keep data time < compute time so the GPU never waits.

---

## Problem

GPU profiling on the EAT frozen encoder training (RTX 6000, 24GB VRAM) with 8
workers showed the GPU was underutilized due to data starvation:

| Metric | 8 workers (initial) |
|---|---|
| GPU Utilization | 84.8% avg, drops to 3% |
| Data Time | 230 ms avg, spikes to 4.7s |
| Compute Time | 477 ms avg |
| Throughput | 481 sam/s |
| VRAM Used | 2.6% (0.66 GB / 25.2 GB) |

The GPU was idle ~15% of the time on average, but with **sharp drops to 3%**
when the prefetch queue drained during data loading spikes.

---

## Pipeline Architecture

### Data flow: disk to GPU

```
┌─────────────────────────── CPU (dataloader workers) ───────────────────────────┐
│                                                                                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  sf.read()   │───>│  Resample    │───>│  Kaldi fbank │───>│  Normalize +  │  │
│  │  (disk I/O)  │    │  32k → 16k   │    │  (128 bins)  │    │  pad/truncate │  │
│  └─────────────┘    └──────────────┘    └──────────────┘    └───────┬───────┘  │
│      ~5-20 ms            ~2-5 ms            ~15-30 ms               │          │
│                                                                     │          │
└─────────────────────────────────────────────────────────────────────┼──────────┘
                                                                      │
                                                            ┌─────────▼─────────┐
                                                            │  DataLoader       │
                                                            │  collate + batch  │
                                                            │  (256 samples)    │
                                                            └─────────┬─────────┘
                                                                      │
                                                            ┌─────────▼─────────┐
                                                            │  pin_memory +     │
                                                            │  CPU → GPU xfer   │
                                                            └─────────┬─────────┘
                                                                      │
┌──────────────────────────── GPU ────────────────────────────────────┼──────────┐
│                                                                     │          │
│  ┌──────────────────┐    ┌────────────────┐    ┌─────────────────┐  │          │
│  │  EAT encoder     │───>│  Mean pool     │───>│  Linear head    │  │          │
│  │  (frozen ViT)    │    │  (skip CLS)    │    │  + BCE loss     │  │          │
│  │  forward only    │    │                │    │  fwd + bwd      │  │          │
│  └──────────────────┘    └────────────────┘    └─────────────────┘  │          │
│      ~400 ms                ~1 ms                  ~70 ms           │          │
│                                                                     │          │
└─────────────────────────────────────────────────────────────────────┘          │
```

### Where was the bottleneck?

```
                        Bottleneck severity (8 workers)
                 ┌──────────────────────────────┐
    sf.read()    │████████░░░░░░░░░░░░░░░░░░░░░░│  ~30% of CPU time
    (disk I/O)   │  Network storage (PACE NFS)   │  Spiky — cache misses
                 ├──────────────────────────────┤
    Resample     │███░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ~10% of CPU time
    32k → 16k    │  torchaudio.functional        │  Fast, negligible
                 ├──────────────────────────────┤
    Kaldi fbank  │██████████████████░░░░░░░░░░░░░│  ~60% of CPU time
    computation  │  torchaudio.compliance.kaldi  │  MAIN BOTTLENECK
                 ├──────────────────────────────┤
    GPU compute  │  Not the bottleneck           │  Only 2.6% VRAM used
    (EAT + head) │  477ms/batch but GPU starved  │  Could handle 10x more
                 └──────────────────────────────┘
```

With 8 workers and batch_size=256, each batch took ~1056ms to produce on CPU
but only ~477ms to consume on GPU — the GPU was idle over half the time.

---

## Solution: Increase to 12 Workers

**PACE CPU:GPU limits (gpu-rtx6000/gpu-v100):**
- Interactive (salloc): 6 CPUs per GPU
- Batch (sbatch): 12 CPUs per GPU

We added `--num-workers` passthrough to `run.sh`/`_job.sh` with automatic
`--cpus-per-task` override, and ran gpu-check comparisons:

### Results: 8 vs 12 workers

| Metric | 8 workers (job 4113823) | 12 workers (job 4113831) |
|---|---|---|
| **GPU Utilization** | 87.3% avg | 85.1% avg |
| **Data Time (avg)** | 61 ms | **5 ms** |
| **Data Time (max)** | 2,523 ms | **1,207 ms** |
| **Compute Time** | 473 ms | 487 ms |
| **Throughput** | 502 sam/s | **523 sam/s** |
| **Data Loading %** | 7.5% | **0.8%** |
| **Job Duration** | 10m 19s | **9m 27s** |

12 workers virtually **eliminated the data bottleneck**: average data time
dropped from 61ms to 5ms, and data loading fell to just 0.8% of step time.

### Timeline: before and after

```
BEFORE (8 workers, data time 61ms avg, spikes to 2.5s):

Workers:  [████ batch N+1 ████████████]  [████ batch N+2 ████████████]
GPU:      [██ compute ██] [░ wait ░]     [██ compute ██] [░ wait ░]
                           ↑ data spike causes GPU idle


AFTER (12 workers, data time 5ms avg):

Workers:  [████ batch N+1 █████]  [████ batch N+2 █████]  (always ahead)
GPU:      [██ compute ██][██ compute ██][██ compute ██]    (nearly continuous)
                          ↑ next batch already prefetched
```

See `outputs/gpu_check/eat-finetune_4113823.png` (8 workers) and
`eat-finetune_4113831.png` (12 workers) for the full utilization plots.

---

## What Are Fbank Features?

Fbank = mel-frequency **f**ilter**bank** features. A standard audio representation:

```
Raw waveform          STFT               Mel filterbank        Fbank output
(80,000 samples)      (windowed FFT)     (triangular filters)  (512 × 128)
                                         on mel frequency
                                         scale

 ╱╲  ╱╲              ┌──────────┐        ┌──────────┐         ┌──────────┐
╱  ╲╱  ╲  ──STFT──>  │▓▓▒▒░░    │ ─mel─> │▓▓▒░      │ ─log─> │ time     │
╲  ╱╲  ╱   25ms      │▓▓▓▒░░   │ scale  │▓▓▒░      │        │ × 128    │
 ╲╱  ╲╱    windows   │▓▓▓▓▒░   │        │▓▓▓▒░     │        │ mel bins │
                     └──────────┘        └──────────┘         └──────────┘
                      freq × time         mel × time          mel × time
```

EAT uses **Kaldi-style** fbanks with specific parameters:
- 128 mel bins, 10ms frame shift, 25ms window, hanning window
- HTK-compatible, no energy, no dither
- Normalization: mean=-4.268, std=4.569

---

## Future: Pre-computing Features to Disk

If 12 workers becomes insufficient (e.g. larger models with faster forward
passes), we can pre-compute features to disk as a one-time step. The key
question is **what** to pre-compute, since different models need different inputs:

| Model | Expected Input | Pre-computable? |
|---|---|---|
| EAT | Kaldi fbank (128 bins, EAT-specific params) | Yes, but EAT-only |
| BEATs | Raw waveform (16kHz) | Just resample |
| AST | Mel spectrogram (128 bins, different params) | Yes, but AST-only |
| wav2vec2 / HuBERT | Raw waveform (16kHz) | Just resample |

**Option A: Pre-compute model-specific features (max speedup, narrow reuse)**

Pre-compute the final input for one model (e.g. EAT fbanks as `.pt` files).
Workers just load tensors — near-zero CPU cost.

```
One-time: ~67k chunks × 33ms ÷ 12 workers ≈ 3 minutes
Per-sample: sf.read + resample + fbank = ~33ms  →  torch.load = ~1-2ms  (16x faster)
Disk cost: ~17 GB (67k × 256KB per tensor)
```

Best when training many epochs with a single model.

**Option B: Pre-compute resampled 16kHz audio (moderate speedup, broad reuse)**

Cache the resampled audio at 16kHz. Removes disk I/O + resampling (~13ms)
but fbank/spectrogram computation still happens per epoch (~20ms).

```
Per-sample: sf.read + resample + fbank = ~33ms  →  load + fbank = ~22ms  (1.5x faster)
Disk cost: ~13 GB (16kHz × 5s × 4 bytes × 67k chunks)
```

Best when comparing multiple model architectures on the same data.

**Option C: Pre-compute mel spectrograms (good balance)**

Mel spectrograms are the most common audio input across models. While EAT
uses Kaldi fbanks and BEATs uses raw waveforms, many other architectures
(AST, PSLA, PANNs) use mel spectrograms. Pre-computing 128-bin mel
spectrograms covers the widest range of models.

```
Per-sample: similar to Option A for compatible models
Disk cost: ~17 GB
```

Best when planning to experiment with multiple spectrogram-based models.

### Current status

With 12 workers, data loading is 0.8% of step time — **not a bottleneck**.
Pre-computing is not needed now but is worth revisiting if we move to faster
models or multi-GPU training where the data pipeline needs to be even faster.
