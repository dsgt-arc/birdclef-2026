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
