---
name: "best-rq-pytorch"
source: "https://github.com/lucidrains/best-rq-pytorch"
license: "MIT"
category: "self-supervised"
pipeline_stage:
  - "tokenization"
  - "representation-learning"
relevance: "Random projection quantization for efficient self-supervised audio learning"
status: "reference"
---

# BEST-RQ PyTorch

## Overview

BEST-RQ (BERT-based Speech pre-Training with Random-projection Quantizer) is a self-supervised learning approach that uses random projections to create discrete audio tokens without learning a codebook. This eliminates codebook collapse issues and provides a simple, stable training procedure. The model can generate semantic tokens suitable for audio language models like SoundStorm.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `BestRQ` | Main model with random projection quantizer | `best_rq_pytorch/` |
| `ConformerWrapper` | Conformer encoder for masked prediction | `best_rq_pytorch/` |
| `BestRQPretrainer` | Training wrapper with masking and loss | `best_rq_pytorch/` |

## Relevance to Pipeline

- **Tokenization**: Alternative to learned codebooks; random projections provide stable, consistent tokenization without training
- **Representation Learning**: Masked prediction pre-training similar to BERT/HuBERT but with simpler target generation
- **Inference**: Trained model can extract semantic tokens for downstream tasks

## Usage

```python
import torch
from best_rq_pytorch import BestRQ, ConformerWrapper, BestRQPretrainer

# Create model with random projection quantizer
brq = BestRQ(
    codebook_size=1024,
    codebook_dim=16,
    sample_rate=24_000,
    n_mels=80,
    win_length=960,
    hop_length=960 // 3,  # 75Hz token rate
    conformer=ConformerWrapper(
        num_tokens=1024,
        conformer=dict(
            dim=1024,
            depth=24,
            heads=16,
            conv_kernel_size=5
        )
    )
)

# Pre-train with masking
trainer = BestRQPretrainer(
    model=brq,
    dataset=...,
    num_train_steps=1_000_000,
    lr=5e-4,
    mask_prob=0.6
)
trainer.train()
```

## Dependencies

- Python 3.x
- PyTorch
- einops
- vector-quantize-pytorch

## Notes

- Key insight: random projections work as well as learned codebooks for self-supervised pre-training
- No codebook collapse issues compared to VQ-VAE approaches
- Can fit k-means on intermediate layer activations for additional quantization
- Used in SoundStorm for semantic token generation
- Simpler than wav2vec 2.0's Gumbel-softmax quantization
