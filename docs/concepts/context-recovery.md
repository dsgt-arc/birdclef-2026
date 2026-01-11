# Concept: Context Recovery from Discrete Audio Tokens

**Status**: developing
**Created**: 2026-01-11
**Updated**: 2026-01-11

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

## Testable Hypotheses

1. **Local Context Dominance**: A 1D CNN (local context) will outperform a simple Attention mechanism (global weighted average) because audio features are locally correlated (e.g., a frequency sweep).
   - *Early Evidence (Exp 002)*: LightCNN1DPooler (~60% acc) > AttentionPooler (~38% acc).
2. **Pareto Optimality**: We can achieve a 20x speedup over the teacher model with less than a 10% drop in accuracy.

## Connections

- **Based on**: `docs/concepts/mlp-mixer-distillation.md` (ideas around simplifying architectures).
- **Related to**: Experiment 002 (BEATs Inference Optimization).
- **Could lead to**: A general-purpose "fast inference" recipe for any token-based audio model (HuBERT, Wav2Vec2).
