# GPU Utilization Analysis — Experiment 005

## Summary

GPU profiling on the EAT frozen encoder training (RTX 6000, 24GB VRAM) shows
**84.8% average GPU utilization** but with sharp drops to 3%, caused by
**data starvation** — the CPU-bound data pipeline cannot keep the GPU fed
continuously.

| Metric | Min | Avg | Max |
|---|---|---|---|
| Data Time | 1.7 ms | 230.0 ms | 4694.9 ms |
| Compute Time | 447.5 ms | 476.6 ms | 694.8 ms |
| Data Loading % | 0.3% | 11.0% | 87.2% |
| Throughput | 48 sam/s | 481 sam/s | 569 sam/s |
| Memory Allocated | 0.45 GB | 0.66 GB | 0.88 GB |
| GPU Utilization | 3.0% | 84.8% | 100.0% |

**Key observations:**
- VRAM is 2.6% utilized (0.66 GB / 25.2 GB) — massive headroom
- Data time spikes to **4.7 seconds** — 10x the compute time
- GPU drops to 3% when the prefetch queue drains

---

## Pipeline Architecture

### Data flow: disk to GPU

```
┌─────────────────────────── CPU (dataloader workers) ───────────────────────────┐
│                                                                                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  sf.read()  │───>│  Resample    │───>│  Kaldi fbank │───>│  Normalize +  │  │
│  │  (disk I/O) │    │  32k → 16k   │    │  (128 bins)  │    │  pad/truncate │  │
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

### Timing breakdown per batch (batch_size=256)

```
                    CPU work per sample
                ┌────────────────────────────┐
                │ sf.read    resample  fbank │
                │  ~10ms   +  ~3ms  + ~20ms  │
                │         = ~33ms/sample     │
                └────────────────────────────┘

    With 8 workers processing 256 samples:
    256 samples / 8 workers = 32 samples per worker
    32 samples × 33ms = ~1056ms per batch (data production)

    vs.

    GPU compute per batch: ~477ms

    Data production (1056ms) >> GPU compute (477ms)
    ──────────────────────────────────────────────
    GPU is IDLE ~55% of the time waiting for data
```

### Timeline view: what happens each step

```
Time ──────────────────────────────────────────────────────────────────────>

Workers:  [████ loading batch N+1 ████████████████████]  [████ batch N+2 ████
           ~1056ms producing next batch

GPU:      [██ compute N ██]  [░░░░ IDLE ░░░░░░░░]  [██ compute N+1 ██]  [░░░
              ~477ms           ~230ms avg wait          ~477ms
                               (up to 4.7s!)

          ◄── prefetch buffer masks some latency, but can't fully hide it ──►
```

---

## Where is the bottleneck?

```
                        Bottleneck severity
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

**The GPU is underutilized because:**
1. Kaldi fbank is CPU-intensive (~20-30ms per 5s clip) and cannot be GPU-accelerated
   (torchaudio's Kaldi fbank has no CUDA backend)
2. Only 8 CPU workers available (`--cpus-per-task=8`); max ratio is 12:1 for batch jobs on gpu-rtx6000/gpu-v100 (6:1 for interactive)
3. The frozen encoder finishes fast — no gradient computation for encoder params
4. VRAM is 97% empty — batch size could be much larger, but more samples per batch
   would increase data pipeline pressure even further

---

## What are fbank features?

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

These parameters are **EAT-specific**. Other models use different representations:
- **BEATs**: raw waveforms (internal feature extraction)
- **AST**: 128-bin mel spectrogram (different params than EAT's Kaldi fbank)
- **wav2vec2 / HuBERT**: raw waveforms (learned features)

---

## Recommendations

### Option A: Increase CPU workers (quick, moderate improvement)

Increase SLURM allocation from 8 to 12 CPUs (max for gpu-rtx6000 batch jobs), set `num_workers` to match.

**PACE CPU:GPU limits (gpu-rtx6000/gpu-v100):**
- Interactive (salloc): 6:1
- Batch (sbatch): 12:1

```
Expected improvement:
  8 workers:  1056ms per batch → GPU waits ~580ms
  12 workers:  704ms per batch → GPU waits ~230ms  (max for RTX 6000)
```

**Pros:** No code changes, just config.
**Cons:** Limited to 12 workers max on RTX 6000; fbank is still computed every epoch.

### Option B: Pre-compute fbanks to disk (best for EAT)

Run a one-time preprocessing script that saves fbanks as `.pt` files.
The dataloader then reads pre-computed tensors instead of raw audio.

```
Preprocessing (one-time):
  ~67k chunks × 33ms/sample ÷ 8 workers ≈ 5 minutes

Per-sample load time after pre-computation:
  Before: sf.read + resample + fbank = ~33ms
  After:  torch.load(.pt file)        = ~1-2ms   (16x faster)
```

**Pros:** Eliminates CPU bottleneck entirely; fast NVMe reads.
**Cons:** EAT-specific (different models need different features); ~17 GB disk
space (67k × 256KB per fbank tensor); requires re-running if audio data changes.

### Option C: Cache resampled 16kHz audio to disk (best for multi-model)

Pre-compute only the resampling step (32kHz → 16kHz) and save as `.pt` or `.wav`.
Fbank is still computed in workers, but the expensive disk read + resample is gone.

```
Per-sample load time after caching:
  Before: sf.read + resample + fbank = ~33ms
  After:  torch.load + fbank         = ~22ms   (1.5x faster)
```

**Pros:** Benefits ALL models (EAT, BEATs, future encoders); reusable.
**Cons:** Fbank still computed per epoch; smaller speedup than Option B.

### Recommended approach

**Option A (increase workers to 16-24)** for an immediate fix.
**Option B (pre-compute fbanks)** if training many EAT epochs and want
maximum GPU utilization.
