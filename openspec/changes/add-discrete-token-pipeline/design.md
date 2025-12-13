# Design: Discrete Token Pipeline

## Context

BirdCLEF is a Kaggle competition for bird species identification from audio recordings. The key constraint is that inference must run on CPU within a strict time budget. The birdclef-2025 approach demonstrated a viable pipeline:

```
Audio → MFCC → K-means tokens → Word2Vec → Logistic Regression
```

This achieved ~0.85 F1 on a subset of species. The 2026 approach aims to improve representation quality while maintaining or improving inference efficiency.

## Goals

1. **Better tokenization**: Birdcall-specific discrete tokens that capture relevant acoustic patterns
2. **Richer representations**: Self-supervised transformer to learn contextualized features
3. **Efficient inference**: Static embeddings that don't require running the transformer at test time
4. **Maximize CPU budget**: Use all available compute for higher quality predictions

## Non-Goals

1. Real-time inference (batch processing is fine)
2. GPU inference (Kaggle uses CPU-only for submission)
3. State-of-the-art accuracy at any cost (efficiency is a hard constraint)
4. Novel research contributions (focus on applying existing techniques)

## Decisions

### Decision 1: Tokenization Approach

**Options considered:**
1. K-means on MFCCs (birdclef-2025 baseline)
2. K-means on mel-spectrograms
3. Random projection (BEST-RQ style)
4. Learned codebook (VQ-VAE, neural codec)

**Decision**: Start with BEST-RQ style random projection as primary approach, with k-means as fallback.

**Rationale**:
- BEST-RQ is simpler (no training), more stable (no codebook collapse), and scales well
- Random projections preserve relative distances (Johnson-Lindenstrauss)
- The birdclef-2025 k-means approach is already working, so we have a fallback
- Neural codecs (WavTokenizer, EnCodec) are trained on speech/music and may not transfer to bird calls

### Decision 2: Self-Supervised Objective

**Options considered:**
1. Contrastive (wav2vec 2.0 style)
2. Reconstruction (predict masked tokens)
3. Iterative clustering (HuBERT style)
4. Online clustering (DinoSR style)

**Decision**: Start with simple reconstruction, iterate to clustering if needed.

**Rationale**:
- Reconstruction is simplest to implement (just cross-entropy on token IDs)
- Codec2Vec showed reconstruction can work well with good tokenization
- Iterative clustering is expensive (multiple training rounds)
- Online clustering adds complexity (teacher-student, EMA)

### Decision 3: Backbone Architecture

**Options considered:**
1. Standard Transformer encoder
2. Mamba (state space model)
3. Efficient Audio Transformer (EAT)

**Decision**: Start with small Transformer encoder, evaluate Mamba if context length is limiting.

**Rationale**:
- Transformer is well-understood, good tooling (Hugging Face, FlashAttention)
- Bird recordings are typically chunked to 5-10 seconds (~50-100 tokens at 10Hz), so quadratic attention is tractable
- Mamba would be useful for full 60+ second recordings, but we can chunk

### Decision 4: Static Embedding Strategy

**Options considered:**
1. Average token embeddings (Word2Vec style)
2. [CLS] token embedding
3. model2vec distillation (mean pool → PCA → learned weights)
4. Attention pooling

**Decision**: Implement model2vec style distillation.

**Rationale**:
- model2vec achieves 90%+ of sentence-transformer quality at much lower cost
- PCA reduces dimensionality without losing much information
- Learned weights can upweight important tokens
- More sophisticated than simple averaging, but still produces static embeddings

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio ──► Mel-Spectrogram ──► Random Projection ──► Tokens │
│                                      ▼                       │
│                              Frozen Codebook                 │
│                                      ▼                       │
│                         Transformer Encoder                  │
│                         (Masked Prediction)                  │
│                                      ▼                       │
│                      model2vec Distillation                  │
│                      (Mean Pool → PCA → Weights)             │
│                                      ▼                       │
│                         Static Embeddings                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio ──► Mel-Spectrogram ──► Tokens ──► Static Embeddings │
│                                                 ▼            │
│                                        Linear Classifier     │
│                                                 ▼            │
│                                          Predictions         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Note: At inference time, we skip the Transformer entirely. The token-to-embedding mapping is a simple lookup table (learned during model2vec distillation).

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Random projections don't work for bird calls | High | Fall back to k-means (proven in 2025) |
| Transformer is too slow to train | Medium | Use small model, leverage existing compute |
| model2vec loses too much information | Medium | Evaluate against full transformer; tune PCA dims |
| Kaggle dataset differs significantly from 2025 | Medium | Design for flexibility; wait for dataset release |

## Open Questions

1. **Dataset**: What species are in BirdCLEF 2026? How much unlabeled soundscape data?
2. **Tokenization rate**: 10Hz (like Codec2Vec) or higher resolution for birds?
3. **Codebook size**: 16k (birdclef-2025) or smaller/larger?
4. **Transformer size**: Base (12 layers) or smaller (6 layers)?
5. **Integration with BirdSet/bioacoustics-model-zoo**: Use their pretrained models as teachers?
