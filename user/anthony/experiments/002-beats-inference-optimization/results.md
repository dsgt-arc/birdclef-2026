# Results: BEATs Inference Optimization

## Part A & B: Profiling and PyTorch Benchmarks

**Date**: 2026-01-10
**Hardware**: NVIDIA L40S GPU, Intel Xeon Gold 6226 CPU (4 threads)

### Profiling Overhead Validation

| Device | Profiled (ms) | Unprofiled (ms) | Overhead |
|--------|---------------|-----------------|----------|
| GPU    | 12.46         | 12.20           | **2.15%** |
| CPU    | 449.92        | 448.43          | **0.33%** |

Both well under the 5% target.

### GPU Benchmarks (L40S)

| Batch | Total (ms) | Throughput (samples/sec) | Peak GPU (MB) |
|-------|------------|--------------------------|---------------|
| 1     | 8.0        | 125                      | 391           |
| 4     | 12.5       | 320                      | 443           |
| 16    | 43.6       | **367**                  | 649           |
| 32    | 91.8       | 349                      | 918           |
| 64    | 183.4      | 349                      | 1457          |

**Optimal batch size**: 16 (best throughput)

#### GPU Stage Breakdown (batch=16)

| Stage       | Time (ms) | Percentage |
|-------------|-----------|------------|
| fbank       | 9.21      | 21.1%      |
| patch_embed | 0.11      | 0.3%       |
| layer_norm  | 0.07      | 0.2%       |
| proj        | 0.16      | 0.4%       |
| transformer | 33.96     | **77.8%**  |
| pool        | 0.03      | 0.1%       |

### CPU Benchmarks (4 threads)

| Batch | Total (ms) | Throughput (samples/sec) |
|-------|------------|--------------------------|
| 1     | 124        | 8.1                      |
| 4     | 447        | 8.9                      |
| 16    | 1852       | 8.6                      |
| 32    | 3876       | 8.3                      |

**CPU throughput is ~constant** regardless of batch size (limited by transformer compute).

#### CPU Stage Breakdown (batch=16)

| Stage       | Time (ms) | Percentage |
|-------------|-----------|------------|
| fbank       | 15.70     | 0.8%       |
| patch_embed | 2.48      | 0.1%       |
| layer_norm  | 0.65      | 0.04%      |
| proj        | 5.93      | 0.3%       |
| transformer | 1826.53   | **98.6%**  |
| pool        | 0.22      | 0.01%      |

### Key Findings

1. **Transformer is the dominant bottleneck**
   - GPU: 77-85% of total time
   - CPU: **98%+** of total time

2. **GPU vs CPU speedup**: ~40x (367 vs 8.6 samples/sec at batch 16)

3. **fbank is significant on GPU** (10-21%) but negligible on CPU (<1%)
   - This suggests ONNX export of frontend will help GPU more than CPU

4. **Memory scales linearly** with batch size on GPU (391 MB → 1457 MB)

### Implications for Static Embeddings

These results strongly validate the static embeddings approach:

- **CPU optimization potential**: Replacing transformer (98% of time) with lookup could yield **50-100x speedup**
- **GPU optimization potential**: More modest gains expected since transformer is "only" 78% of time
- **Frontend ONNX export**: Will provide additional speedup, especially on GPU where fbank is 21%

### Artifacts

- Benchmark results: `~/scratch/birdclef-2026/outputs/exp002-benchmarks/pytorch_benchmarks.json`
- Profiling module: `user/anthony/src/birdclef/profiling.py`

---

## Part C: ONNX Export

**Date**: 2026-01-10
**Hardware**: Intel Xeon Gold 6226 CPU (4 threads)

### Export Details

| Model | Output Matches | Max Diff | ONNX File Size |
|-------|----------------|----------|----------------|
| Frontend | Yes | 4.77e-06 | 2.2 MB |
| Full Encoder | Yes | 2.86e-06 | 362 MB |

Both models exported with dynamic batch size and sequence length axes using legacy exporter (`dynamo=False`).

### Frontend ONNX Benchmarks (fbank → patch embeddings)

| Batch | ONNX (ms) | PyTorch (ms) | Speedup |
|-------|-----------|--------------|---------|
| 1     | 0.65      | 2.27         | **3.52x** |
| 4     | 2.47      | 7.05         | **2.85x** |
| 16    | 10.61     | 24.76        | **2.33x** |
| 32    | 22.27     | 48.12        | **2.16x** |

### Full Encoder ONNX Benchmarks

| Batch | ONNX (ms) | PyTorch (ms) | Speedup |
|-------|-----------|--------------|---------|
| 1     | 121.90    | 124.09       | 1.02x |
| 4     | 491.57    | 447.08       | 0.91x |
| 16    | 2073.52   | 1851.58      | 0.89x |
| 32    | 4282.69   | 3876.25      | 0.91x |

### Key Findings

1. **Frontend ONNX excels** - 2-3.5x speedup for the simple Conv2D + projection operations

2. **Full encoder ONNX is slightly slower** - PyTorch's native CPU transformer optimizations are superior
   - This is expected: ONNX Runtime's generic graph executor can't match PyTorch's specialized attention kernels

3. **Validates static embeddings approach**:
   - Fast ONNX frontend (~0.65ms for single sample) for feature extraction
   - Lookup table replaces the slow transformer entirely
   - Combined pipeline could achieve 50-100x CPU speedup

### Artifacts

- Frontend ONNX: `~/scratch/birdclef-2026/outputs/exp002-benchmarks/onnx/beats_frontend.onnx`
- Full encoder ONNX: `~/scratch/birdclef-2026/outputs/exp002-benchmarks/onnx/beats_encoder.onnx`
- Benchmark results: `~/scratch/birdclef-2026/outputs/exp002-benchmarks/onnx/onnx_benchmarks.json`

---

## Part D: Static Embeddings Distillation

*Pending...*

## Part E: Static Embeddings Inference

*Pending...*

## Part F: Analysis

*Pending...*
