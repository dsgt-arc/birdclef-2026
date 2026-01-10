# Proposal: BEATs Inference Profiling and Optimization

**Status**: proposed
**Created**: 2026-01-10
**Author**: anthony

## Related Documentation

- **Concepts**:
  - [MLP-Mixer Distillation](../../../docs/concepts/mlp-mixer-distillation.md) - Broader distillation strategy
- **References**:
  - [Model2Vec](../../../docs/vendor/model2vec.md) - Static embedding technique
  - [BEATs](../../../docs/vendor/beats.md) - Model architecture
- **Prior Experiments**:
  - [001-encoder-benchmark](../001-encoder-benchmark/results.md) - BEATs achieves 95.8% on ESC-50 (107s eval time on GPU)

## Hypothesis

BEATs inference can be dramatically accelerated using model2vec-style static embeddings while maintaining acceptable accuracy. We expect:
1. Static embeddings achieve **100x speedup** over full BEATs with **<5% accuracy drop**
2. The transformer is the dominant cost (>70% of total time) - confirming static embeddings target the right bottleneck
3. ONNX Runtime provides ≥2x speedup over PyTorch on CPU for full model

Understanding the split times validates the model2vec approach.

## Background

From experiment 001, BEATs achieves 95.8% on ESC-50 with 107s total evaluation time on GPU. However, we don't know:
- Where time is spent (fbank vs patch embedding vs transformer vs pooling)
- How performance scales with batch size
- CPU vs GPU tradeoffs for different batch sizes
- Whether ONNX provides meaningful speedups

For BirdCLEF, we have a compute budget ceiling - being significantly under budget wastes potential. This experiment profiles the pipeline and applies model2vec-style optimization to bypass the transformer.

### Tokenizer-Guided Model2Vec Approach

We leverage the BEATs Tokenizer's learned semantic clustering to build our codebook, then create static embeddings that bypass the transformer at inference.

**Distillation (one-time):**
1. Run BEATs **Encoder** on corpus → 768-dim patch embeddings (transformer input)
2. Run BEATs **Tokenizer** on corpus → discrete token IDs (0-1023)
3. Pair each patch embedding with its token ID
4. Average patch embeddings per token ID → codebook (1024 × 768)
5. Pass each centroid through Encoder transformer (single-patch) → static embeddings

**Inference (fast):**
1. **Frontend** (fbank + patch_embed + proj): ONNX, produces 768-dim patches
2. **Quantize**: Nearest neighbor to codebook centroids → token IDs
3. **Lookup**: Index into static embeddings (1024 × 768)
4. **Pool**: Mean over patches → final embedding

**Why this is smart:** The Tokenizer learned which spectral patterns are semantically equivalent. We use its clustering to build a codebook in the input space, avoiding blind k-means on spectral features.

See `design.md` for detailed pipeline diagrams.

## Method

### Approach

1. Instrument BEATs pipeline with split-time profiling
2. Benchmark full model across device × runtime × batch size configurations
3. Build patch codebook and static embeddings (model2vec)
4. Benchmark static embedding pipeline
5. Compare accuracy vs speedup tradeoffs

### Pipeline Stages to Profile

```
Audio Waveform
    ↓ [Stage 1: fbank]
Mel Filterbank (T × 128)
    ↓ [Stage 2: patch_embed]
Patch Embeddings (T' × 512)
    ↓ [Stage 3: transformer]  ← TARGET FOR MODEL2VEC
Contextualized Features (T' × 768)
    ↓ [Stage 4: pool]
Final Embedding (768,)
```

### Benchmark Matrix

| Config | Device | Runtime | Model | Batch Sizes |
|--------|--------|---------|-------|-------------|
| A | GPU | PyTorch | Full BEATs | 1, 4, 16, 32, 64 |
| B | CPU | PyTorch | Full BEATs | 1, 4, 16, 32 |
| C | CPU | ONNX | Full BEATs | 1, 4, 16, 32 |
| D | CPU | ONNX + NumPy | Static (model2vec) | 1, 4, 16, 32 |

### Setup

| Component | Details |
|-----------|---------|
| Data | ESC-50 (2k clips, 5s each @ 16kHz) |
| GPU | V100-16GB |
| CPU | HPC node (record cores/model) |
| Dependencies | BEATs, onnxruntime, torch |
| Code | New profiling module in `src/birdclef/` |

### Procedure

**Part A: Instrumentation**
1. Add timing hooks to BEATs pipeline stages
2. Implement warmup + N timed iterations protocol
3. Record per-stage latency and memory

**Part B: PyTorch Benchmarks**
1. GPU: Run all batch sizes, record split times
2. CPU: Run all batch sizes, record split times
3. Measure peak memory for each configuration

**Part C: ONNX Export**
1. Export tokenizer (fbank + patch_embed) to ONNX
2. Export full BEATs to ONNX (handle dynamic axes)
3. Run CPU benchmarks with ONNX Runtime
4. Compare to PyTorch CPU baseline

**Part D: Static Embeddings (Tokenizer-Guided Model2Vec)**
1. Load BEATs Tokenizer checkpoint (Tokenizers_iter3.pt)
2. For each audio in ESC-50 corpus:
   - Extract 768-dim patch embeddings (Encoder frontend)
   - Extract discrete token IDs (Tokenizer full pipeline)
   - Store (patch_embedding, token_id) pairs
3. Build codebook: average patch embeddings per token ID → (1024 × 768)
4. Compute static embeddings: pass each centroid through Encoder transformer (single-patch)
5. Save codebook.npy + embeddings.npy
6. Implement fast inference: ONNX frontend → quantize → lookup → pool

**Part E: Static Embedding Benchmarks**
1. Benchmark config D across batch sizes
2. Run ESC-50 5-fold CV for accuracy comparison
3. Compare accuracy vs speedup tradeoff

**Part F: Analysis**
1. Plot throughput vs batch size curves (all configs)
2. Create split time breakdown charts
3. Plot accuracy vs speedup Pareto frontier
4. Identify optimal operating points

### Variables

- **Independent**:
  - Device (GPU vs CPU)
  - Runtime (PyTorch vs ONNX vs ONNX+NumPy)
  - Model (Full BEATs vs Static embeddings)
  - Batch size (1, 4, 16, 32, 64)
- **Dependent**:
  - Latency per sample (ms)
  - Throughput (samples/sec)
  - Split times per stage (%)
  - Peak memory (MB)
  - Accuracy (for static embeddings)
  - Token coverage (how many of 1024 tokens used)
- **Controlled**: Base model (BEATs iter3), Tokenizer (iter3), dataset (ESC-50), codebook size (1024)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Latency | Time per sample at given batch size (ms) |
| Throughput | Samples processed per second |
| Split time | % of total time per pipeline stage |
| Memory | Peak GPU/RAM usage (MB) |
| Efficiency | Throughput / peak memory |

### Baseline

From experiment 001 (batch_size=1, GPU, PyTorch):
| Metric | Value |
|--------|-------|
| Total time | 107s for 2000 samples |
| Throughput | ~19 samples/sec |
| Accuracy | 95.8% |

### Expected Outcomes

| Config | Model | Expected Throughput | Expected Accuracy |
|--------|-------|---------------------|-------------------|
| A (GPU bs=1) | Full | ~19 samples/sec | 95.8% |
| A (GPU bs=32) | Full | ~100+ samples/sec | 95.8% |
| B (CPU bs=1) | Full | ~2 samples/sec | 95.8% |
| C (CPU ONNX bs=16) | Full | ~10 samples/sec | 95.8% |
| D (CPU Static bs=16) | Static | ~200+ samples/sec | ~91% |

### Success Criteria

- **Confirm if**:
  1. Static embeddings achieve ≥100x speedup vs CPU PyTorch (config B)
  2. Static embeddings accuracy ≥91% (≤5% relative drop from 95.8%)
  3. Transformer accounts for >50% of full model compute time
  4. ONNX provides ≥2x speedup over PyTorch on CPU
- **Reject if**:
  1. Static embeddings speedup <10x (tokenizer dominates cost)
  2. Static embeddings accuracy <86% (>10% relative drop)
  3. Profiling overhead >5% or variance >10%

## Deliverables

1. **Profiling module**: Reusable instrumentation for BEATs
2. **ONNX exports**: Tokenizer and full model
3. **Static embedding artifacts**: Codebook + embedding matrix
4. **Benchmark results**: JSON/CSV with all configurations
5. **Visualization**: Throughput curves, split times, accuracy vs speedup
6. **Recommendations**: Optimal settings for different compute budgets

## Limitations

- ESC-50 clips are 5s; real BirdCLEF clips vary in length
- Single GPU type (V100); results may differ on other hardware
- ONNX export may not support all BEATs features
- Codebook quality depends on corpus diversity
- Static embeddings lose position-dependent information
- Profiling adds some overhead (mitigate with warmup)

## Future Work (out of scope for this experiment)

- Quantization (INT8/FP16) of static embeddings
- TensorRT optimization
- Learned codebook (end-to-end training)
- Variable codebook sizes
- Multi-GPU inference
