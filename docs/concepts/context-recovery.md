# Concept: Context Recovery from Discrete Audio Tokens

**Status**: developing
**Created**: 2026-01-11
**Updated**: 2026-01-25

## Idea

The core idea is to decouple **representation learning** (what a sound is) from **contextual modeling** (how sounds evolve over time).

We utilize a high-quality, pre-trained audio tokenizer (like the BEATs tokenizer) to generate a sequence of discrete tokens or static patch embeddings. Instead of processing these with a heavy Transformer (as in the original model) or averaging them into a "bag-of-features" (as in Model2Vec), we introduce a **lightweight "Context Recovery" module** (e.g., a 1D CNN or shallow Attention layer) to recapture essential temporal relationships and sequence grammar.

## Motivation

Our experiments with static embeddings (Experiment 002) revealed a critical trade-off:
- **Full BEATs Model**: High accuracy (84% on ESC-50) but slow (150ms/sample).
- **Static Embeddings (Bag-of-Words)**: Ultra-fast (4ms/sample) but poor accuracy (36%).

The accuracy collapse (48% drop) confirms that **temporal context is non-negotiable** for bioacoustics. A bird call is defined not just by its frequencies (the tokens) but by their order (the syntax).

This concept aims to find the "Goldilocks" zone: achieving **>80% of the teacher's accuracy** while maintaining **>90% of the static pipeline's speed**.

## Background

- **BEATs Tokenizer**: A VQ-VAE style tokenizer that maps audio patches to discrete codes.
- **Model2Vec**: A technique for distilling sentence transformers into static word embeddings. We attempted to adapt this for audio patches.
- **Context Collapse**: The phenomenon where averaging or independently processing sequence tokens destroys the information contained in their order.

## Open Questions

1. **How much context is local vs. global?** Do we need a Transformer's global attention, or is a Conv1D's local receptive field sufficient to recognize a chirp?
2. **What is the optimal lightweight architecture?** CNNs, RNNs (LSTM/GRU), or Linear Attention?
3. **Does this scale to long audio?** Current experiments use 5-second clips. Can a lightweight pooler handle minute-long recordings efficiently?
4. **Can we use sparse distillation targets?** Do we need to match all 248 teacher positions, or is a subset sufficient?

## Testable Hypotheses

1. **Local Context Dominance**: A 1D CNN (local context) will outperform a simple Attention mechanism (global weighted average) because audio features are locally correlated (e.g., a frequency sweep).
   - *Early Evidence (Exp 002)*: LightCNN1DPooler (~60% acc) > AttentionPooler (~38% acc).
2. **Pareto Optimality**: We can achieve a 20x speedup over the teacher model with less than a 10% drop in accuracy.
3. **Sparse Target Sufficiency**: Matching 25-50% of teacher embedding positions provides equivalent learning signal to matching 100%.

---

## Sparse Distillation Targets

### The I/O Bottleneck Problem

During Experiment 003 (Model Distillation), we encountered significant I/O bottlenecks when training on cached teacher embeddings. The cache for AudioSet-500k is ~350GB (v2 format), requiring sophisticated solutions:

- **ShardAffinityDataset**: Worker-shard affinity to eliminate redundant I/O
- **Two-level prefetching**: Background copy to local NVMe + memory prefetch
- **v2 cache format**: Removed static_embeds (reconstructed at runtime via lookup table)

See: `docs/references/streaming-cache-implementation.md`

Despite these optimizations, I/O remains a limiting factor. Compression (zstd) provides only ~8% savings because fp16 embeddings don't compress well at the bit level.

### Insight: Semantic Compression via Sparsity

EAT (Efficient Audio Transformer) demonstrates that audio spectrograms are highly redundant:

> "We feed only the unmasked patches to the encoder... this reduces the sequence length and computation complexity quadratically."

EAT masks 70-80% of patches during pre-training and still achieves BEATs-level performance in 15x less training time. This proves that **20-30% of patches contain sufficient information** to learn robust representations.

### Applying Sparsity to Distillation

Instead of storing all 248 teacher embedding positions, we could store only 25-50%:

| Format | Positions | Size per sample | Total (500k) |
|--------|-----------|-----------------|--------------|
| v2 (current) | 248 | ~381 KB | ~350 GB |
| Sparse 50% | 124 | ~191 KB | ~175 GB |
| Sparse 25% | 62 | ~96 KB | ~88 GB |

The I/O reduction is proportional to the mask ratio.

### Key Differences from EAT

| Aspect | EAT Pre-training | Sparse Distillation |
|--------|------------------|---------------------|
| What's sparse | Encoder input | Distillation targets |
| Loss type | Predict MASKED patches | Match VISIBLE patches |
| Compute savings | Yes (sparse encoder) | No (full student forward) |
| I/O savings | No | Yes (2-4x reduction) |

EAT's loss forces reconstruction of masked patches. Our sparse distillation would only match visible anchors. This is a weaker learning signal, but may be sufficient due to:

1. **ConvNeXt1D's receptive field**: Each output depends on ~379 input positions. You can't "cheat" at position 50 without getting neighboring positions right.
2. **Temporal smoothness**: Audio representations are locally correlated. Matching sparse anchors implicitly constrains intermediate positions.

### Implementation Options

**Option 1: Training-time masking (no cache change)**
```python
# Randomly subsample at each batch
mask = torch.randperm(248)[:62]  # 25% of positions
loss = F.mse_loss(student[:, mask], teacher[:, mask])
```
- Pro: No cache regeneration, easy to test
- Con: No I/O savings (still load full cache)

**Option 2: Fixed grid masking**
```python
# Keep every 4th position: [0, 4, 8, 12, ...]
indices = torch.arange(0, 248, 4)  # 62 positions
```
- Pro: No index storage needed, deterministic
- Con: Less diversity, might miss some patterns

**Option 3: Random per-sample masking (new cache format)**
```
shard.safetensors:
  teacher_embeds: (N, 62, 768)  # Sparse
  token_ids: (N, 248)           # Full (for input)
  sparse_indices: (N, 62)       # Which positions were kept
```
- Pro: Maximum diversity, full I/O savings
- Con: Requires cache regeneration, index storage overhead

### Auxiliary Losses for Sparse Training

To compensate for the weaker learning signal, consider auxiliary losses:

**Temporal smoothness:**
```python
# Encourage coherent representations between adjacent positions
loss_smooth = F.mse_loss(student[:, 1:], student[:, :-1])
```

**Consistency regularization:**
```python
# Two forward passes with different masks should produce similar representations
student_out1 = student(input)
student_out2 = student(input)  # Different dropout/augmentation
loss_consist = F.mse_loss(student_out1.mean(1), student_out2.mean(1))
```

### Validation Strategy

1. **Phase 1**: Training-time masking with varying ratios (100%, 50%, 25%)
   - Compare final ESC-50 accuracy
   - If 25% matches 100%, sparse cache is validated

2. **Phase 2**: If Phase 1 succeeds, regenerate cache in sparse format
   - 4x I/O reduction
   - Faster epoch times

### Expected Outcomes

| Mask Ratio | Cache Size | Accuracy Impact (hypothesis) |
|------------|------------|------------------------------|
| 0% (full) | 350 GB | Baseline |
| 50% | 175 GB | <2% drop |
| 75% | 88 GB | <5% drop |

The hypothesis is that ConvNeXt1D's inductive bias (local context via dilated convolutions) naturally interpolates between sparse anchor points.

---

## Connections

- **Based on**: `docs/concepts/mlp-mixer-distillation.md` (ideas around simplifying architectures).
- **Related to**: Experiment 002 (BEATs Inference Optimization).
- **Related to**: `docs/vendor/eat.md` (EAT's sparse encoding validates redundancy in audio).
- **Motivated by**: `docs/references/streaming-cache-implementation.md` (I/O bottleneck).
- **Could lead to**: A general-purpose "fast inference" recipe for any token-based audio model (HuBERT, Wav2Vec2).
