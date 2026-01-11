# Concept: Distilled Sequence Model for Context Recovery

**Status**: developing
**Created**: 2026-01-11
**Updated**: 2026-01-11

## Idea

Instead of training a lightweight context recovery module (like a 1D CNN) from scratch on labeled data (ESC-50), we propose **distilling the full BEATs Transformer into a lightweight sequence model** using unlabeled data.

The student model would take **discrete tokens** (or their corresponding static embeddings) as input and learn to predict the **teacher's contextualized embeddings** (or logits).

## Motivation

1.  **Unlimited Training Data**: We are currently limited by the size of labeled datasets like ESC-50 (2,000 samples). By using distillation, we can train on **any audio** (e.g., the entire BirdCLEF training set, Bioacoustics Model Zoo) since we only need the teacher's output as the target.
2.  **Rich Supervision**: The teacher (BEATs) provides a "dense" supervision signal (768-dimensional vectors per time step or soft probability distributions) which is much richer than a single hard class label.
3.  **Efficiency**: We keep the "fast path" of `Audio -> Tokenizer -> Static Lookup` and replace the heavy BEATs Transformer with a tiny, optimized student.

## Architecture Candidates

The student model must be significantly faster than the 12-layer BEATs Transformer while capturing sufficient temporal context.

1.  **Tiny Transformer**: A 2-4 layer, 256-dim Transformer.
    *   *Pros*: Same architecture as teacher, easy to distill.
    *   *Cons*: Still $O(N^2)$ attention, though $N$ is small (248 patches).
2.  **MLP-Mixer / ConvMixer**: An all-MLP or Conv-based architecture.
    *   *Pros*: $O(N)$ linear complexity, very fast on CPU.
    *   *Cons*: Limited global context compared to attention (but maybe sufficient for 5s clips).
3.  **State Space Models (Mamba/S4)**:
    *   *Pros*: Linear complexity, excellent sequence modeling.
    *   *Cons*: Newer, potentially harder to optimize/deploy on standard ONNX runtimes.

## Training Pipeline

1.  **Data**: Unlabeled audio (e.g., BirdCLEF 2026 training data, Bioacoustics Model Zoo).
2.  **Teacher (Frozen)**: Full BEATs Model.
    *   Input: Raw Audio / Spectrogram.
    *   Output: Sequence of 768-d contextualized embeddings (before pooling).
3.  **Student (Trainable)**:
    *   Input: Discrete Token Sequence (from BEATs Tokenizer).
    *   Lookup: Frozen Codebook (from Exp 002).
    *   Backbone: Lightweight Sequence Model (e.g., Tiny Transformer).
    *   Output: Predicted Sequence of Embeddings.
4.  **Loss Function**:
    *   **MSE / Cosine Similarity**: Match the teacher's per-patch embeddings.
    *   **KL Divergence**: Match the teacher's output probability distribution (if using logits).

## Inference Pipeline

```
Audio -> BEATs Tokenizer -> Discrete Tokens (Indices)
           |
           v
      Static Lookup (Fast) -> Uncontextualized Embeddings
           |
           v
      Distilled Student -> Contextualized Embeddings
           |
           v
      Classifier / Pooler -> Prediction
```

## Connections

- **Extends**: `docs/concepts/context-recovery.md` (adds distillation and unlabeled data).
- **Related to**: `docs/concepts/mlp-mixer-distillation.md` (shares the goal of efficient inference).
- **Implements**: The "Contextual Pooler" idea but with a much stronger training signal.
