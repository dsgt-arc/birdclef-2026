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

**Date**: 2026-01-10
**Dataset**: ESC-50 (2000 samples, 5 seconds each)

### Extraction Summary

| Metric | Value |
|--------|-------|
| Total patches extracted | 496,000 |
| Patches per sample | 248 |
| Token vocabulary size | 1,024 |
| Tokens used in ESC-50 | 932 (91%) |
| Unused tokens | 92 |

### Token Distribution

Highly skewed Zipf-like distribution:

| Percentile | Patches per Token |
|------------|-------------------|
| 10th | 158 |
| 25th | 247 |
| 50th (median) | 387 |
| 75th | 609 |
| 90th | 862 |
| 99th | 1,914 |
| Max (Token 172) | 50,997 |

**Token 172 dominates**: 10.3% of all patches map to a single token. This validates the need for SIF weighting.

### Model2Vec Enhancements

1. **PCA Dimensionality Reduction**
   - Input: 768 dimensions
   - Output: 256 dimensions
   - Explained variance: **98.5%** (excellent retention)
   - Whitening: enabled

2. **SIF (Smooth Inverse Frequency) Weighting**
   - Weight range: [0.0096, 0.9802]
   - Formula: w = a / (a + p) where a=1e-3
   - Heavily downweights Token 172 (0.0096 weight vs 0.98 for rare tokens)

### Artifacts Saved

| File | Shape | Description |
|------|-------|-------------|
| codebook.npy | (1024, 768) | Averaged patch embeddings per token |
| static_embeddings.npy | (1024, 768) | Transformer-processed centroids |
| static_embeddings_pca.npy | (1024, 256) | PCA-reduced embeddings |
| sif_weights.npy | (1024,) | SIF weights per token |
| pca_model.pkl | - | Fitted sklearn PCA model |
| token_stats.json | - | Distribution statistics |

**Location**: `~/scratch/birdclef-2026/outputs/exp002-benchmarks/static_embeddings/`

---

## Part E: Static Embeddings Inference

**Date**: 2026-01-11
**Hardware**: Intel Xeon Gold 6226 CPU (4 threads)

### Initial Results: Naive Quantization (BROKEN)

| Method | Batch 1 (ms) | Throughput (samples/sec) |
|--------|--------------|--------------------------|
| Full BEATs (PyTorch) | 147.14 | 6.8 |
| Static (naive) | 392.43 | 2.5 |

**Problem**: Static embeddings was 2.7x SLOWER than full BEATs due to naive L2 distance computation creating 776 MB intermediate tensors per sample.

### Fix 1: scipy cdist

Replaced naive broadcasting with `scipy.spatial.distance.cdist`:

| Method | Batch 1 (ms) | Throughput | vs BEATs |
|--------|--------------|------------|----------|
| Full BEATs CPU | 149.9 | 6.7/sec | baseline |
| cdist 768d | 84.1 | 11.9/sec | **1.8x** |

**Result**: Fixed the regression, now 1.8x faster than BEATs. But cdist is still the bottleneck (~84ms of 85ms total).

### Fix 2: FAISS IndexFlatL2

Replaced cdist with FAISS for SIMD-optimized nearest neighbor search:

| Method | Batch 1 (ms) | Throughput | vs BEATs |
|--------|--------------|------------|----------|
| Full BEATs CPU | 149.9 | 6.7/sec | baseline |
| cdist 768d | 84.1 | 11.9/sec | 1.8x |
| **FAISS 768d** | **4.6** | **217/sec** | **32.5x** |

### Final Benchmark Results

**Speedup vs Full BEATs CPU (higher is better):**

| Method | Batch 1 | Batch 4 | Batch 16 | Batch 32 |
|--------|---------|---------|----------|----------|
| cdist 768d | 1.8x | 1.5x | 1.5x | 1.6x |
| cdist 768d + SIF | 1.7x | 1.5x | 1.5x | 1.6x |
| cdist 256d + SIF | 1.7x | 1.5x | 1.5x | 1.6x |
| **FAISS 768d** | **32.5x** | **31.2x** | **30.2x** | **30.9x** |
| **FAISS 768d + SIF** | **33.0x** | **29.7x** | **29.0x** | **29.3x** |

**Raw timings (ms):**

| Method | Batch 1 | Batch 4 | Batch 16 | Batch 32 |
|--------|---------|---------|----------|----------|
| Full BEATs CPU | 149.9 | 514.6 | 2051.6 | 4423.5 |
| cdist 768d | 84.1 | 335.6 | 1357.0 | 2728.3 |
| FAISS 768d | 4.6 | 16.5 | 68.0 | 143.2 |

**Throughput (samples/sec):**

| Method | Batch 1 | Batch 4 | Batch 16 | Batch 32 |
|--------|---------|---------|----------|----------|
| Full BEATs CPU | 6.7 | 7.8 | 7.8 | 7.2 |
| cdist 768d | 11.9 | 11.9 | 11.8 | 11.7 |
| **FAISS 768d** | **217.0** | **242.6** | **235.4** | **223.5** |

### Why FAISS is 18x Faster than cdist

1. **SIMD vectorization**: Uses AVX2/AVX-512 for parallel distance computation
2. **Optimized memory access**: Cache-friendly access patterns
3. **Purpose-built**: Specifically designed for nearest neighbor search

### Key Findings

1. **FAISS achieves 32x speedup** over full BEATs on CPU
2. **217 samples/sec** on CPU approaches GPU speed (367 samples/sec)
3. **Practical implications**:
   - 100k audio files: BEATs CPU ~4 hours → FAISS static ~8 minutes
   - No GPU required for fast inference

### Hypothesis Validation

Original hypothesis: "100x speedup with <5% accuracy drop"

- **Speedup achieved**: 32x (not 100x, but significant)
- **Accuracy**: Pending ESC-50 evaluation

The 100x target was overly optimistic. The 32x speedup comes from:
- Eliminating transformer (98% of CPU time)
- FAISS SIMD-optimized quantization

### Artifacts

- Benchmark results: `~/scratch/birdclef-2026/outputs/exp002-benchmarks/static_inference_benchmarks.json`
- Pipeline implementations: `user/anthony/src/birdclef/static_embeddings.py`
  - `StaticEmbeddingsPipeline` (cdist-based)
  - `StaticEmbeddingsPipelineFAISS` (FAISS-based)

---

## Part F: ESC-50 Accuracy Evaluation

*In progress...*
