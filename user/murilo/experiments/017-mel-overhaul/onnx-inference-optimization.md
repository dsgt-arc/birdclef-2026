# ONNX Inference Optimization for Mel Overhaul Models

## Problem

The mel resolution upgrade from (n_fft=1024, n_mels=128) to (n_fft=4096, n_mels=224)
introduced a severe inference bottleneck on Kaggle CPU. The original v4a submission
took **>1 hour** to process the test soundscapes (vs ~38min for v1 at n_fft=1024), risking
timeout on the 2-hour Kaggle limit.

The root cause was the **per-segment mel computation** approach: for each 5-second
competition segment, the kernel independently loaded an audio chunk and computed a full
STFT. With 10-second inference windows (2.5s context padding) and TTA shifts (center,
-2.5s, +2.5s), this meant **36 independent STFT computations per 60-second soundscape**
— each with n_fft=4096, which is 4x more expensive than n_fft=1024. Most of these
windows overlapped significantly, making the redundant computation wasteful.

### Per-segment approach (before optimization)

The original `predict_soundscape()` flow for each soundscape file:

```
For each 5s segment (12 per 60s file):
    For each TTA shift (center, -2.5s, +2.5s):
        1. sf.read() — load 10s audio chunk from disk
        2. librosa.melspectrogram() — compute full STFT (n_fft=4096) + mel filterbank
        3. Normalize (log + AudioSet stats)
        4. ONNX session.run() — single-sample inference (1, 1, 224, 1000)
        5. sigmoid → probabilities
    Average TTA predictions (50%/25%/25%)
Post-process: temperature → smoothing → TopN → metadata priors
```

**Cost per soundscape**: 12 segments × 3 TTA shifts = **36 disk reads + 36 STFTs + 36 ONNX calls**

The STFTs dominated wall time because n_fft=4096 requires 4x the frequency bins (2049
vs 513), and each operates on 320,000 samples (10s × 32kHz) independently.

## Solution

Three optimizations were implemented in commit `7be6362` (2026-03-25), inspired by the
[2025 1st place solution](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md)
which computes a full-signal spectrogram and slices frames rather than computing
independent per-segment STFTs.

### 1. Full-signal mel spectrogram (biggest win)

Instead of computing an independent STFT for each overlapping segment, compute the mel
spectrogram **once** for the entire 60-second file, then slice frames for each
segment/TTA window using numpy indexing (near-free).

**Key insight**: The STFT is a linear operation on overlapping audio frames. Computing it
once over the full signal and slicing the resulting time-frequency representation gives
mathematically identical results to computing it on each audio chunk independently — as
long as the normalization is compatible (see normalization note below).

```
After: 1 sf.read() + 1 STFT for 60s padded audio + 36 numpy slices (near-free)
```

Implementation in `submission.py`:

- `compute_full_signal_mel(audio, mel_cfg)` — computes the mel for an entire audio
  signal (e.g., 60s padded). Returns raw normalized mel as `(n_mels, total_frames)`
  without batch/channel dims or frame truncation. For `audioset` normalization
  (log + global mean/std), this is mathematically equivalent to per-segment computation
  because the normalization uses fixed statistics (not per-window).

- `_slice_mel_window(full_mel, start_frame, target_frames)` — slices a `(1, 1, n_mels,
  target_frames)` window from the full mel, with zero-padding for windows that extend
  beyond file boundaries (needed for TTA shifts at the start/end of the file).

The audio is padded on both sides before mel computation:
```python
pad_samples = context_samples + tta_shift_samples  # 2.5s context + 2.5s TTA
audio_padded = np.pad(audio, (pad_samples, pad_samples), mode="constant")
```
An `origin_frame` offset tracks where t=0 falls in the padded mel, so all segment
positions are computed relative to the original audio timeline.

**Normalization note**: For `audioset` normalization (fixed mean/std from training set),
full-signal mel is mathematically equivalent to per-segment. For `amplitude_to_db`
normalization (per-sample standardize + min-max), there is a subtle difference: training
normalizes per 10-second window, but full-signal normalization operates on the entire
60-second file. In practice, the 2025 1st place solution uses the same full-signal
approach with their normalization, and LB verification confirmed no score regression.

### 2. Batched ONNX inference

Instead of running `session.run()` once per mel slice (36 calls per file), concatenate
all mel slices into a single batch and run **one** batched inference call per file.

```python
mel_batch = np.concatenate([s[2] for s in all_slices], axis=0)  # (36, 1, 224, 1000)
all_probs = _run_onnx_batch(session, mel_batch, mel_cfg)        # (36, 234)
```

This amortizes ONNX session overhead (graph execution setup, memory allocation) and
enables ONNX Runtime's internal parallelism to process multiple samples concurrently.
The ONNX model already supports dynamic batch size via `dynamic_axes={"mel": {0: "batch"}}`.

The TTA averaging is then done as a weighted sum over the batch results:

```python
probs_matrix = np.zeros((n_segments, n_classes), dtype=np.float32)
for i, (seg_idx, weight, _) in enumerate(all_slices):
    probs_matrix[seg_idx] += weight * all_probs[i]
```

### 3. ONNX session optimization

The default ONNX Runtime session uses single-threaded execution with minimal graph
optimization. Three session options were added:

```python
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = os.cpu_count() or 4
sess_opts.inter_op_num_threads = 1
```

- **`ORT_ENABLE_ALL`**: Enables all graph optimizations — operator fusion (e.g., fusing
  Conv+BN+ReLU), constant folding, and redundant node elimination. Reduces the number
  of ops executed per inference call.

- **`intra_op_num_threads = os.cpu_count()`**: Parallelizes computation within individual
  operators (e.g., matrix multiplications, convolutions) across all available CPU cores.
  Kaggle kernels typically have 4 CPU cores.

- **`inter_op_num_threads = 1`**: Single execution stream — avoids thread contention
  between independent operators that would compete for the same CPU cores. Since we're
  already using all cores within each op, multi-stream execution would cause
  over-subscription.

## Activation logic

The full-signal path is automatically selected for SED models with standard mel
normalization:

```python
use_fullsignal = mel_cfg.get("architecture") == "sed" and mel_type == "mel"
```

This routes to `predict_soundscape_fullsignal()` instead of the original per-segment
`predict_soundscape()`. The per-segment path remains available for non-SED models (EAT,
ProtoCLR, ConvNeXt) that use different mel types (kaldi_fbank, protoclr_mel) where the
full-signal approach isn't implemented.

Since the v4b-v4e ablations, the full-signal path also handles `amplitude_to_db`
normalization via a branch in `compute_full_signal_mel()`.

## Results

| Submission | Inference time | LB score |
|------------|---------------|----------|
| v4a (original, per-segment) | >1 hour | 0.825 |
| v4a (optimized, full-signal) | <32 min | 0.826 |
| v1 (per-segment, n_fft=1024) | ~38 min | 0.795 |

The optimized v4a is **~2.5-3x faster** than the original per-segment approach and
**faster than v1** despite 1.75x more mel bins. The ±0.001 LB variance between the
two v4a submissions is expected — Kaggle rotates/subsamples test data between
submissions to prevent overfitting. The optimization is lossless.

All v4b/v4d/v4e submissions automatically use the optimized path.

## Reference

The full-signal mel approach was inspired by the
[BirdCLEF 2025 1st place inference notebook](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md):

> **Full-signal method**: Extract one long spectrogram, then slice into overlapping
> windows to preserve spectral continuity.

Their solution also uses:
- n_fft=4096, 224 mel bins, hop_length=1252 (vs our 320)
- AmplitudeToDB normalization with 80dB top_db + standardize + min-max [0,1]
- 20-second inference windows (vs our 10-second)

We adapted the full-signal concept but kept our own mel parameters, normalization scheme,
and TTA strategy. The batched ONNX inference and session optimization were our additions.

## Files

| File | Description |
|------|-------------|
| `010-.../kaggle_kernel/submission.py` | `compute_full_signal_mel()`, `_slice_mel_window()`, `predict_soundscape_fullsignal()`, `_run_onnx_batch()`, session options |
| `010-.../scripts/export_efficientnet_sed.py` | ONNX export with `dynamic_axes` for batch dim |

## Git history

| Commit | Description |
|--------|-------------|
| `7be6362` | Introduced all three optimizations (full-signal mel, batched ONNX, session options) |
| Uncommitted | Extended `compute_full_signal_mel()` with `amplitude_to_db` normalization branch for v4b support |
