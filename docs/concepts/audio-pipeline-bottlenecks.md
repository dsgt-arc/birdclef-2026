# Concept: Audio Pipeline Bottlenecks — Disk I/O vs GPU Mel Computation

**Status**: developing
**Created**: 2026-02-16
**Updated**: 2026-02-16

## Idea

When building GPU-accelerated audio pipelines with nnAudio, the intuition is
that moving mel spectrogram computation to GPU eliminates the CPU bottleneck.
In practice, the real bottleneck is **disk I/O and CPU resampling** during
audio loading — not mel computation. nnAudio solves the wrong bottleneck
unless data loading is also optimized.

A naive `waveform_only` DataLoader that reads audio files on every
`__getitem__` call starves the GPU, producing spiky ~0% utilization with
brief bursts to ~80%. The GPU finishes its forward pass faster than workers
can feed the next batch.

## Motivation

Discovered during experiment 004 (frame-based tokenization). After integrating
nnAudio for GPU mel computation, GPU utilization on a V100 was terrible despite
the mel transform running on-device. The root cause was `sf.read()` +
`librosa.resample()` costing ~190ms per sample in `BirdSetDataset.__getitem__`.

## Background

### The Audio Loading Pipeline

```
Disk → sf.read() → librosa.resample() → torch.Tensor → GPU → nnAudio mel → ViT
       ~50ms          ~140ms              ~0.1ms        transfer   ~2ms      ~10ms
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       THIS is the bottleneck (190ms/sample)
                                                                   ^^^^^^^^^^
                                                                   NOT this
```

### Per-Sample Cost Breakdown (HSN dataset, PACE cluster)

| Operation         | Time/sample | Where |
|-------------------|-------------|-------|
| `sf.read()`       | ~50ms       | CPU (disk I/O) |
| `librosa.resample()` | ~140ms  | CPU (kaiser_best) |
| Pad/crop          | ~0.1ms      | CPU |
| nnAudio mel       | ~2ms        | GPU |
| ViT forward       | ~10ms       | GPU |

The CPU audio loading is **~10x slower** than the entire GPU pipeline.

### Evidence

HSN pre-loading timings from exp 004:
- Train (5,460 samples): 1,025s → **188ms/sample**
- Val (12,000 samples): 528s → **44ms/sample** (less resampling needed)

The train split takes 4x longer per sample than val, likely because train
files have varying sample rates requiring `librosa.resample()`, while val
files are mostly at 32kHz already.

### Anthony's Approach (Experiments 008, 009)

Anthony's experiments avoid this bottleneck through:
1. **Partial reads**: `sf.read(path, start=X, frames=Y)` — reads only the
   needed chunk, not the entire file
2. **Manifest caching**: pre-computes file metadata (duration, sample rate)
   to JSON, avoiding repeated `sf.info()` calls
3. **Recording-sequential sampling**: chunks from the same recording are
   accessed sequentially for better OS page cache locality
4. **Higher prefetch**: `prefetch_factor=8`, `num_workers=8`

## Caching Strategies

### 1. Cache waveforms in RAM (current approach)

Pre-load all audio into a contiguous `torch.Tensor` during `setup()`.
Zero disk I/O during training.

- **Pros**: Simple, zero training overhead, works with any DataLoader config
- **Cons**: Slow startup (~25 min for HSN), high RAM usage (~10 GB for HSN)
- **Best for**: Small-to-medium datasets that fit in memory, long training runs

### 2. Save cached tensors to `.pt` files (planned optimization)

After first pre-load, save to `HSN_train_waveforms.pt`. Subsequent runs
do `torch.load()` in ~5 seconds.

- **Pros**: One-time cost, instant subsequent loads
- **Cons**: Requires disk space (~10 GB), invalidated if preprocessing changes
- **Best for**: Repeated runs on the same dataset

### 3. On-the-fly with high parallelism

Use many DataLoader workers with aggressive prefetching to overlap I/O
with GPU computation. No pre-loading needed.

- **Pros**: No startup delay, low memory
- **Cons**: May still starve GPU if I/O is too slow per sample
- **Best for**: Large datasets that don't fit in memory, fast storage (NVMe)

### 4. HuggingFace Arrow-cached decoded audio

Configure HF datasets to decode audio into Arrow format. Subsequent
accesses are memory-mapped from the Arrow cache.

- **Pros**: Managed by HF, no custom code
- **Cons**: Requires initial decode pass, Arrow cache can be large
- **Best for**: HF dataset workflows

## Open Questions

- Does the BirdSet HF dataset store audio as decoded arrays or file paths?
  (Currently file paths — `sample["audio"]["path"]` with `bytes=None`)
- Can we avoid `librosa.resample()` entirely by pre-resampling the dataset
  to 32kHz once and caching?
- What is the break-even point where on-the-fly loading (strategy 3) is
  faster than pre-loading (strategy 1) for a given dataset size and epoch count?
- Would `torchaudio.transforms.Resample` (GPU) be faster than `librosa.resample`
  (CPU) for the resampling step?

## Testable Hypotheses

1. Saving cached waveforms to `.pt` files reduces startup from ~25 min to <10s
   with no change in training throughput.
2. Pre-resampling the dataset to 32kHz eliminates the ~140ms/sample resampling
   cost, making on-the-fly loading viable without caching.
3. GPU resampling via `torchaudio.transforms.Resample` inside the model
   (alongside nnAudio mel) would eliminate both disk I/O and CPU resampling
   bottlenecks if combined with raw-audio caching.

## Connections

- **Based on**: Experiment 004 GPU utilization debugging
- **Related to**: `docs/concepts/temporal-resolution-sequence-length.md`,
  Anthony's experiments 008 (attention-pooled SED) and 009 (BirdCLEF pipeline)
- **Could lead to**: `.pt` waveform caching in DataModule, shared pre-resampled
  dataset on project storage, GPU resampling module

## Notes

- The 4x difference in per-sample loading time between train (188ms) and val
  (44ms) strongly suggests `librosa.resample()` is the dominant cost, not
  `sf.read()`. The val split likely has files already at 32kHz.
- For PACE specifically, the HF cache lives on the home directory or scratch
  filesystem. NVMe-backed `$TMPDIR` would be faster for audio reads, but the
  HF cache is not there by default.
- The waveform caching approach (strategy 1) is essentially what
  `cache_mel_specs` did before, but storing raw waveforms (~640KB/sample)
  instead of mel spectrograms (~256KB/sample). The trade-off is ~2.5x more
  RAM for the ability to use GPU mel computation.
