# Design: BEATs Inference Optimization

## Context

BEATs achieves excellent accuracy (95.8% on ESC-50) but inference is slow. For BirdCLEF, we have a compute budget ceiling - being under budget wastes potential. We need to understand where time is spent and optimize accordingly.

The model2vec technique from NLP achieves 500x speedups by replacing transformer inference with static embedding lookup. We adapt this for audio by treating patch embeddings as the "vocabulary" to be mapped to static embeddings.

## Goals

- Profile BEATs pipeline to identify bottlenecks
- Achieve 100x speedup over CPU baseline using static embeddings
- Maintain ≥91% accuracy (≤5% relative drop)
- Produce reusable ONNX artifacts for deployment

## Non-Goals

- Training new models (we use pretrained BEATs)
- Quantization or TensorRT (future work)
- Multi-GPU inference
- Real-time streaming inference

## High-Level Pipelines

### Pipeline A: Full BEATs (Baseline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL BEATS PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio Waveform (16kHz)                                             │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   fbank     │  torchaudio.compliance.kaldi.fbank()               │
│  │  (Stage 1)  │  Output: (T, 128) mel spectrogram                  │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ patch_embed │  Conv2d(1, 512, kernel=patch_size, stride=patch)   │
│  │  (Stage 2a) │  Output: (N_patches, 512)  [N = T' × F']           │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │  layer_norm │  LayerNorm(512)                                    │
│  │  (Stage 2b) │                                                    │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   proj      │  Linear(512, 768)  [post_extract_proj]             │
│  │  (Stage 2c) │  Output: (N_patches, 768)                          │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ transformer │  12-layer TransformerEncoder                       │
│  │  (Stage 3)  │  Output: (N_patches, 768) contextualized           │
│  └─────────────┘  ← BOTTLENECK (expected >70% of compute)           │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │    pool     │  Mean pooling over patches                         │
│  │  (Stage 4)  │  Output: (768,) final embedding                    │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  Final Embedding → Linear Classifier → Prediction                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline B: Static Embeddings (Tokenizer-Guided Model2Vec)

```
┌─────────────────────────────────────────────────────────────────────┐
│              TOKENIZER-GUIDED STATIC EMBEDDING PIPELINE             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ONE-TIME DISTILLATION (uses both models)        │    │
│  │                                                              │    │
│  │  For each audio in corpus:                                   │    │
│  │                                                              │    │
│  │  1. Run BEATs ENCODER (stop before transformer)              │    │
│  │     Audio → fbank → patch_embed → layer_norm → proj          │    │
│  │     Result: patch_embeddings (N_patches × 768)               │    │
│  │                                                              │    │
│  │  2. Run BEATs TOKENIZER (full pipeline)                      │    │
│  │     Audio → fbank → ... → transformer → quantize             │    │
│  │     Result: token_ids (N_patches,) in range [0, 1023]        │    │
│  │                                                              │    │
│  │  3. Pair each patch with its semantic label                  │    │
│  │     (patch_embedding, token_id) for all patches              │    │
│  │                                                              │    │
│  │  After processing corpus:                                    │    │
│  │                                                              │    │
│  │  4. Build codebook by averaging patches per token ID         │    │
│  │     For token_id in 0..1023:                                 │    │
│  │       codebook[token_id] = mean(patches where label==token_id)│    │
│  │     Result: codebook (1024 × 768)                            │    │
│  │                                                              │    │
│  │  5. Compute static embeddings (model2vec style)              │    │
│  │     Pass each codebook centroid through BEATs ENCODER        │    │
│  │     transformer as single-patch sequence                     │    │
│  │     Result: static_embeddings (1024 × 768)                   │    │
│  │                                                              │    │
│  │  KEY INSIGHT: Tokenizer provides semantic clustering;        │    │
│  │  we build fast lookup in input space aligned to it.          │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   FAST INFERENCE                             │    │
│  │                                                              │    │
│  │  Audio Waveform (16kHz)                                      │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌─────────────┐                                             │    │
│  │  │   fbank     │  (ONNX) torchaudio fbank                    │    │
│  │  └─────────────┘                                             │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌─────────────┐                                             │    │
│  │  │ patch_embed │  (ONNX) Conv2d + LayerNorm + Linear         │    │
│  │  │ + norm+proj │  Output: (N_patches, 768)                   │    │
│  │  └─────────────┘                                             │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌─────────────┐                                             │    │
│  │  │  quantize   │  Nearest neighbor to codebook (L2 distance) │    │
│  │  └─────────────┘  Output: (N_patches,) token IDs [0-1023]    │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌─────────────┐                                             │    │
│  │  │   lookup    │  Index into static_embeddings (NumPy)       │    │
│  │  └─────────────┘  Output: (N_patches, 768)                   │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌─────────────┐                                             │    │
│  │  │    pool     │  Mean pooling over patches                  │    │
│  │  └─────────────┘  Output: (768,) final embedding             │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  Final Embedding → Linear Classifier → Prediction            │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Comparison

| Stage | Full BEATs | Static Embeddings |
|-------|------------|-------------------|
| fbank | PyTorch/ONNX | ONNX |
| patch_embed | PyTorch/ONNX | ONNX |
| layer_norm | PyTorch/ONNX | ONNX |
| post_extract_proj | PyTorch/ONNX | ONNX |
| transformer | **12-layer transformer** | **Skipped (lookup instead)** |
| quantize | N/A | Nearest neighbor to 1024 centroids |
| lookup | N/A | NumPy indexing into static embeddings |
| pool | Mean | Mean |

## Decisions

### Decision 1: Codebook Construction

**Options considered:**
1. Blind k-means on patch embeddings (8192 centroids)
2. Use BEATs Tokenizer labels to guide clustering (1024 centroids)
3. Use Tokenizer's VQ codebook directly (wrong space - 256-dim)

**Decision**: Tokenizer-guided clustering (Option 2)

**Rationale**: The Tokenizer learned semantically meaningful clusters during pre-training. By using its labels, our codebook aligns with learned semantic structure rather than just spectral similarity. Size of 1024 matches Tokenizer's vocabulary.

### Decision 2: Codebook Space

**Options considered:**
1. 768-dim (post_extract_proj output, transformer input)
2. 512-dim (patch_embed output, pre-projection)
3. 256-dim (Tokenizer's quantize space)

**Decision**: 768-dim (Option 1)

**Rationale**: This is the space the transformer operates in. Building codebook here means we quantize exactly what would go into the transformer.

### Decision 3: Static Embedding Computation

**Options considered:**
1. Single-centroid through transformer (true model2vec)
2. Centroid with synthetic context
3. Average of contextualized embeddings from corpus

**Decision**: Single-centroid through transformer (Option 1)

**Rationale**: Tests the model2vec hypothesis directly. If it fails, we learn that audio transformers need context more than text transformers.

### Decision 4: Tokenizer Export

**Options considered:**
1. Export fbank + patch_embed + proj together to ONNX
2. Keep fbank in torchaudio, export only patch_embed + proj
3. Reimplement fbank in NumPy

**Decision**: Export fbank + patch_embed + proj together to ONNX

**Rationale**: Single ONNX model simplifies deployment. Need everything up to (but not including) transformer.

## Artifacts

```
artifacts/
├── benchmarks/
│   ├── pytorch_gpu_results.json
│   ├── pytorch_cpu_results.json
│   ├── onnx_cpu_results.json
│   └── static_cpu_results.json
├── models/
│   ├── beats_encoder_frontend.onnx  # fbank + patch_embed + proj (no transformer)
│   ├── beats_full.onnx              # full model for comparison
│   └── static_embeddings/
│       ├── codebook.npy             # (1024, 768) centroids from Tokenizer labels
│       └── embeddings.npy           # (1024, 768) static embeddings
├── distillation/
│   ├── patch_token_pairs.npz        # (patch_embedding, token_id) pairs from corpus
│   └── token_counts.json            # how many patches per token ID
└── figures/
    ├── throughput_vs_batch.png
    ├── split_times.png
    ├── accuracy_vs_speedup.png
    └── token_distribution.png       # histogram of token usage
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quantization error degrades accuracy | High | Try larger codebook, position-aware variants |
| Tokenizer dominates runtime | High | Would invalidate model2vec approach; profile first |
| ONNX export fails for fbank | Medium | Fall back to torchaudio + ONNX patch_embed |
| Codebook doesn't generalize | Medium | Build from diverse corpus, test on held-out data |
| Position information lost | Medium | Try position-weighted pooling |

## How This Relates to BEATs Tokenizer

We USE the BEATs Tokenizer during distillation, but our final inference pipeline is different:

| Aspect | BEATs Tokenizer | Our Distillation | Our Inference |
|--------|-----------------|------------------|---------------|
| Quantizes | Transformer OUTPUT | Uses Tokenizer labels | Transformer INPUT |
| Codebook dim | 256-dim | 768-dim (input space) | 768-dim |
| Codebook size | 1024 | 1024 (same) | 1024 |
| Requires transformer | Yes (to produce tokens) | Yes (one-time) | **No** |

The key insight: we use the Tokenizer's semantic knowledge to BUILD our codebook, but at inference we don't need the Tokenizer - just nearest-neighbor lookup in the 768-dim input space.

## Open Questions

- Does single-patch transformer inference retain useful information without attention?
- Are all 1024 token IDs well-represented in ESC-50, or are some rare/unused?
- Does position information matter for ESC-50? (5s clips, environmental sounds)
- How much does corpus choice affect codebook quality?
- Should we filter out rare tokens (< N examples) from codebook?

---

## Part G: Contextual Pooling (Recovery from Context Collapse)

### Problem Statement

The static embeddings approach (Part E-F) achieved 32x speedup but lost 48% accuracy.
Root cause: **Context collapse** - passing single patches through the transformer destroys cross-patch attention relationships.

Current pipeline:
```
patches[248, 768] → lookup static_embeddings → mean pool → 768d output
```

The mean pooling discards all sequential/contextual information.

### Solution: Learnable Contextual Pooling

Add a lightweight learnable layer AFTER the static embedding lookup to recover contextual information:

```
patches[248, 768] → lookup static_embeddings[248, 768] → CONTEXTUAL POOLER → 768d output
```

### Option 1: Attention Pooling

Simple learnable weighted average over patches.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ATTENTION POOLING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Static Embeddings: (batch, 248, 768)                               │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Linear    │  Linear(768 → 1)                                   │
│  │  (768 → 1)  │  Output: (batch, 248, 1) attention logits          │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Softmax   │  Softmax over sequence dim                         │
│  │   (dim=1)   │  Output: (batch, 248, 1) attention weights         │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ Weighted Sum│  (embeddings * weights).sum(dim=1)                 │
│  └─────────────┘  Output: (batch, 768) pooled embedding             │
│                                                                     │
│  Parameters: 768 + 1 = 769                                          │
│  FLOPs per sample: ~248 × 768 × 2 ≈ 380K                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Minimal parameters (769)
- Very fast inference
- Learns which patches are important

**Cons:**
- No local pattern detection
- No cross-patch interaction

### Option 2: 1D CNN Pooling

Captures local patterns in the patch sequence.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         1D CNN POOLING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Static Embeddings: (batch, 248, 768)                               │
│         │                                                           │
│         ▼  transpose to (batch, 768, 248) for Conv1d                │
│  ┌─────────────┐                                                    │
│  │   Conv1d    │  Conv1d(768, 768, kernel=3, padding=1)             │
│  │  (layer 1)  │  + BatchNorm1d(768) + ReLU                         │
│  └─────────────┘  Output: (batch, 768, 248)                         │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Conv1d    │  Conv1d(768, 768, kernel=3, padding=1)             │
│  │  (layer 2)  │  + BatchNorm1d(768) + ReLU                         │
│  └─────────────┘  Output: (batch, 768, 248)                         │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ Global Pool │  AdaptiveAvgPool1d(1) or Mean                      │
│  └─────────────┘  Output: (batch, 768)                              │
│                                                                     │
│  Parameters: 2 × (768 × 768 × 3 + 768 × 2) ≈ 3.5M                   │
│  FLOPs per sample: ~2 × 248 × 768 × 768 × 3 ≈ 880M                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Captures local patterns (kernel=3 sees 3 adjacent patches)
- More expressive than attention pooling
- Translation invariant

**Cons:**
- More parameters (3.5M)
- Slower than attention pooling
- May overfit on small datasets

### Option 3: Lightweight CNN (Reduced Channels)

Compromise between expressivity and efficiency.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LIGHTWEIGHT 1D CNN POOLING                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Static Embeddings: (batch, 248, 768)                               │
│         │                                                           │
│         ▼  transpose to (batch, 768, 248)                           │
│  ┌─────────────┐                                                    │
│  │   Conv1d    │  Conv1d(768, 256, kernel=3, padding=1)             │
│  │  (project)  │  + ReLU                                            │
│  └─────────────┘  Output: (batch, 256, 248)                         │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Conv1d    │  Conv1d(256, 256, kernel=3, padding=1)             │
│  │  (layer 1)  │  + ReLU                                            │
│  └─────────────┘  Output: (batch, 256, 248)                         │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Conv1d    │  Conv1d(256, 768, kernel=1)                        │
│  │  (expand)   │  Output: (batch, 768, 248)                         │
│  └─────────────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ Global Pool │  Mean over sequence                                │
│  └─────────────┘  Output: (batch, 768)                              │
│                                                                     │
│  Parameters: 768×256×3 + 256×256×3 + 256×768×1 ≈ 980K               │
│  FLOPs per sample: ~248 × (768×256×3 + 256×256×3 + 256×768) ≈ 290M  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Protocol

Both poolers are trained end-to-end on ESC-50:

1. **Freeze**: Static embeddings lookup (no gradients)
2. **Train**: Contextual pooler + linear classifier
3. **Data**: ESC-50 5-fold CV (same as baseline)
4. **Optimizer**: Adam, lr=1e-3
5. **Epochs**: 100 (same as baseline)
6. **Batch size**: 32

### Expected Outcomes

| Approach | Parameters | Inference (ms) | Expected Accuracy | Notes |
|----------|------------|----------------|-------------------|-------|
| Mean pool (baseline) | 0 | 4.6 | 35.9% | Current |
| Attention pool | 769 | ~5 | 50-60%? | Learns importance |
| Full CNN (768→768) | 3.5M | ~15 | 70-80%? | May recover context |
| Light CNN (768→256→768) | 980K | ~8 | 65-75%? | Balance |

### Success Criteria

- **Minimum**: Recover to 70% accuracy (halve the gap to BEATs 84%)
- **Target**: Reach 80% accuracy with <10ms inference
- **Stretch**: Match BEATs 84% accuracy

### Comparison to Full BEATs

| Aspect | Full BEATs | Static + Attention | Static + CNN |
|--------|------------|-------------------|--------------|
| Accuracy | 84% | 50-60%? | 70-80%? |
| Inference (ms) | 150 | ~5 | ~15 |
| Speedup | 1x | 30x | 10x |
| Parameters | 90M | 769 | 980K-3.5M |
| Trainable | 0 (frozen) | pooler only | pooler only |

### Implementation Plan

1. Create `ContextualPooler` base class with `forward(embeddings) → pooled`
2. Implement `AttentionPooler(ContextualPooler)`
3. Implement `CNN1DPooler(ContextualPooler)`
4. Modify `StaticEmbeddingsPipelineFAISS` to accept optional pooler
5. Create training script for ESC-50 with pooler
6. Run 5-fold CV for each pooler variant
7. Benchmark inference speed for each variant
