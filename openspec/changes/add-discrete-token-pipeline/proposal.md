# Change: Add Discrete Token Pipeline for BirdCLEF 2026

## Why

The birdclef-2025 approach demonstrated that discrete tokenization + Word2Vec embeddings can achieve competitive classification performance with efficient inference. However, the current pipeline has limitations:

1. **Tokenization**: K-means on MFCCs is domain-agnostic; birdcall-specific tokenization could capture acoustic patterns better
2. **Representation Learning**: Word2Vec is shallow; modern self-supervised transformers learn richer contextualized representations
3. **Static Embeddings**: Averaging Word2Vec vectors loses sequence information; model2vec-style distillation could preserve more signal
4. **Inference**: The pipeline isn't optimized for the Kaggle CPU budget constraint

## What Changes

### Phase 1: Birdcall-Specific Tokenization
- Research and implement birdcall-optimized audio codecs or clustering approaches
- Compare: k-means on MFCCs (baseline), k-means on mel-spectrograms, learned codebook (VQ-VAE style), random projections (BEST-RQ style)
- Evaluate codebook utilization and reconstruction quality

### Phase 2: Self-Supervised Transformer
- Train a masked prediction transformer on discrete tokens (Codec2Vec-style)
- Explore target strategies: reconstruction, iterative clustering (HuBERT), online clustering (DinoSR)
- Evaluate representation quality via linear probing

### Phase 3: Static Embeddings via model2vec
- Distill transformer representations into static token embeddings
- Use the model2vec methodology: mean pooling → PCA → learned weighting
- Goal: fixed-dimension embedding per audio clip without running the transformer

### Phase 4: Efficient Inference Pipeline
- Simple linear classifier on static embeddings
- Optimize for Kaggle CPU budget (maximize predictions within time limit)
- Ensemble strategies if compute allows

## Impact

- Affected specs: New capabilities (tokenization, self-supervised-learning, static-embeddings, inference)
- Affected code: Builds on `vendor/birdclef-2025/birdclef/mel2vec/` patterns
- Dependencies: PyTorch, faiss, transformers (or custom), model2vec
